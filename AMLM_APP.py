"""
mini_matlab.py

A self-contained PyQt5 application that implements a "mini-MATLAB" for:
- CSV/Excel data import
- Nonlinear fitting (user-defined model)
- Nonlinear equation solving (single equation, single variable or system)
- Plotting (Matplotlib embedded)
- Export: fitted parameters (CSV/XLSX), figure (PNG/SVG), project save/load (JSON)

Requirements (put in requirements.txt):
PyQt5
numpy
scipy
matplotlib
pandas
sympy
openpyxl

Run:
pip install -r requirements.txt
python mini_matlab.py

To build an EXE (Windows):
pyinstaller --onefile --windowed mini_matlab.py

Notes:
- The user model input should use 'x' as independent variable and parameter names like a,b,c (comma-separated in the Params field).
- For safety, user expressions are parsed via sympy and lambdified.
- If you import an Excel file, ensure openpyxl is installed.

"""

import sys
import json
import os
import traceback
import time
from functools import partial

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QListWidget, QLineEdit, QTextEdit, QMessageBox, QComboBox,
    QSplitter, QFormLayout, QGroupBox, QSpinBox
)



import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import root
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import sympy as sp


class CallCounter:
    """简单调用计数器：用于统计模型函数和雅可比函数的调用次数。"""
    def __init__(self, func):
        self.func = func
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.func(*args, **kwargs)


def estimate_lm_jac_calls(model_calls, param_dim):
    """
    对 SciPy LM 的 jac_calls 做理论估计。
    在未显式提供 jac 的情况下，LM 通常采用有限差分近似雅可比，
    每一轮大致需要 1 次基准残差计算 + p 次参数扰动计算。

    因此：
        iterations_est ≈ ceil(model_calls / (p + 1))
        jac_calls_est ≈ p * iterations_est
    """
    p = max(int(param_dim), 0)
    mc = max(int(model_calls), 0)
    if p == 0 or mc == 0:
        return 0
    iterations_est = int(np.ceil(mc / (p + 1)))
    return p * iterations_est


def benchmark_runtime(func, repeat=5, warmup=1):
    """
    更稳妥的计时函数：
    - 先 warmup，减少首次调用带来的缓存/初始化抖动
    - 再重复多次，取最小值作为核心求解 runtime
    """
    repeat = max(int(repeat), 1)
    warmup = max(int(warmup), 0)

    for _ in range(warmup):
        func()

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)

    return float(min(times)) if times else float('nan')


# ============================================================================
#                               工具函数区域
# ============================================================================
# -------------------- Utility functions --------------------

def parse_model(expr_str, param_names):
    """（解释：这是“模型表达式解析器”）
    作用：
        将用户输入的数学表达式（字符串）转换为可供 numpy+curve_fit 使用的 Python 函数。

    例如：
        expr_str: 'a*exp(-b*x)+c'
        param_names: ['a','b','c']

    返回：一个可以调用的函数 f(x, *params)
    """
    x = sp.symbols('x')
    #创建一个数学符号x，并把这个符号的数值传递给变量x（所以这里的x并不相同，一个是数学符号，一个是变量名）

    params = sp.symbols(' '.join(param_names)) if param_names else []
    #当传递过来的参数param_names里有内容时，把里面的内容使用空格的方式连接到一起，这是' '.join()语句的功能，连接到一起后传递给参数param


        
    # 使用 sympy 把字符串转成数学表达式
    try:
        expr = sp.sympify(expr_str, convert_xor=True)
        #sympify把人写的数学公式转化成计算机能理解的内容
        #convert_xor=True因为在python里^符号表示异或，但有时用户会使用这个符号表示幂运算，这里就是告诉程序，如果用户写了^那就是幂运算
    except Exception as e:
        raise ValueError(f"无法解析表达式: {e}")
        #except就是当try语句出现问题时，程序主动报错，让用户自查，而不是跳到什么未知页面去


    # 把 sympy 表达式转换成 numpy 可用的函数
    try:
        func = sp.lambdify((x,)+params, expr, modules=['numpy'])
        """举例：
           expr = a * sp.sin(x) + b * x**2
           func = sp.lambdify((x,) + params, expr, modules=['numpy'])
           print(func(1.0, 2.0, 3.0))  # 输出: 2*sin(1) + 3*1^2 ≈ 4.68
        """
    except Exception as e:
        raise ValueError(f"无法转换为数值函数: {e}")
    
    # curve_fit 需要的包装格式
    def wrapped(x_array, *pvals):
        # ensure numpy array
        x_arr = np.asarray(x_array)
        return np.asarray(func(x_arr, *pvals))

    return wrapped


def build_jacobian_sympy(expr_str, param_names):
    """构建模型函数及其对参数的雅可比矩阵（符号求导 + 数值化）。
    返回:
        model(x, *theta) -> y_pred
        jac(x, *theta)   -> J (N×P)
    """
    x = sp.symbols('x')
    if not param_names:
        raise ValueError("参数名为空，无法构建雅可比矩阵。")
    params = sp.symbols(' '.join(param_names))
    expr = sp.sympify(expr_str, convert_xor=True)

    # 对每个参数求偏导，得到一组符号表达式
    J_syms = [sp.diff(expr, p) for p in params]

    f = sp.lambdify((x,) + params, expr, modules=['numpy'])
    Jf = sp.lambdify((x,) + params, J_syms, modules=['numpy'])

    def model(x_array, *theta):
        x_arr = np.asarray(x_array, dtype=float)
        return np.asarray(f(x_arr, *theta), dtype=float)

    def jac(x_array, *theta):
        x_arr = np.asarray(x_array, dtype=float)
        # Jf 返回长度为P的结果；其中某些偏导（例如对常数项 c 的偏导）可能返回标量 1，
        # 需要广播成与 x_arr 同长度的向量，避免 vstack 维度不一致。
        vals = Jf(x_arr, *theta)

        cols = []
        for v in vals:
            arr = np.asarray(v, dtype=float)
            if arr.ndim == 0:
                # 标量 -> (N,)
                arr = np.full_like(x_arr, float(arr), dtype=float)
            else:
                arr = arr.reshape(-1)
                if arr.size == 1 and x_arr.size > 1:
                    # 单元素向量 -> (N,)
                    arr = np.full_like(x_arr, float(arr[0]), dtype=float)
            if arr.shape[0] != x_arr.shape[0]:
                raise ValueError(f"雅可比列长度不匹配：期望 {x_arr.shape[0]}，得到 {arr.shape[0]}")
            cols.append(arr)

        # 组成 (N, P)
        J = np.column_stack(cols)
        return J
    return model, jac


def amlm_fit(x, y, model_func, jacobian_func, theta0, max_iter=200, tol=1e-8):
    """AMLM（Adaptive/Modified Levenberg–Marquardt）拟合。
    与标准 LM 的区别：阻尼矩阵使用 D = diag(JᵀJ)，并用信任域比率 ρ 自适应调节 λ。

    返回:
        theta       (最优参数)
        rss         (残差平方和)
        ok          (是否收敛/成功)
        iters       (迭代次数)
        model_calls (模型函数调用次数)
        jac_calls   (雅可比函数调用次数)
        eval_cost   (统一代价 = model_calls + jac_calls)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    theta = np.asarray(theta0, dtype=float).copy()
    lam = 1.0
    ok = True

    # 简单数据清洗：剔除 NaN/Inf（避免线性代数崩溃导致“闪退”）
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return theta, np.nan, False, 0, 0, 0, 0

    for k in range(int(max_iter)):
        y_pred = model_func(x, *theta)
        r = (y_pred - y).reshape(-1, 1)          # (N,1)
        J = jacobian_func(x, *theta)             # (N,P)

        # 梯度与近似 Hessian
        g = (J.T @ r).ravel()                    # (P,)
        H = (J.T @ J)                            # (P,P)

        # D = diag(JᵀJ)，并做一个很小的下界，避免 D 出现 0 导致矩阵病态
        d = np.diag(H).copy()
        d[d <= 1e-12] = 1e-12
        D = np.diag(d)

        # 解 (H + λD) δ = -g
        A = H + lam * D
        b = -g
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 退化情况：用最小二乘兜底，尽量不让程序崩溃
            delta, *_ = np.linalg.lstsq(A, b, rcond=None)

        if not np.all(np.isfinite(delta)):
            ok = False
            break

        theta_new = theta + delta
        r_new = (model_func(x, *theta_new) - y).reshape(-1, 1)

        S = 0.5 * float((r.T @ r))
        S_new = 0.5 * float((r_new.T @ r_new))

        # 预测下降量（LM/信任域比率常用形式）
        pred = 0.5 * float(delta.T @ (lam * (D @ delta) - g))

        # 如果预测下降量非正，扩大 λ 重新来
        if not np.isfinite(pred) or pred <= 0:
            lam *= 2.0
            continue

        rho = (S - S_new) / pred

        if rho > 0:
            theta = theta_new
            # 根据 ρ 调整 λ
            if rho > 0.75:
                lam *= 0.5
            elif rho < 0.25:
                lam *= 2.0
        else:
            lam *= 2.0

        if np.linalg.norm(delta) < tol:
            break

    rss = float(np.sum((model_func(x, *theta) - y) ** 2))
    model_calls = int(getattr(model_func, 'calls', 0))
    jac_calls = int(getattr(jacobian_func, 'calls', 0))
    eval_cost = model_calls + jac_calls
    return theta, rss, ok, k + 1, model_calls, jac_calls, eval_cost



    #为什么要curve_fit一下？因为用户输入的公式里的参数abc没有具体数值，需要通过x和y一组组的数据把abc的值拟合出来，要想拟合，就需要转换成curve_fit能识别的形式
    #pvals指代所有有可能输入的参数们





def estimate_parameter_std_errors(x, y, theta, model_func, jacobian_func):
    """基于最优点处的局部线性近似，估计参数标准误差。
    协方差近似为 sigma^2 * (J^T J)^(-1)，其中 sigma^2 = RSS / (n - p)。
    若矩阵病态，则使用伪逆；若样本数不足或计算失败，则返回 NaN。
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float).ravel()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = x.size
    p = theta.size
    if n == 0 or p == 0 or n <= p:
        return np.full(p, np.nan, dtype=float)

    try:
        y_fit = np.asarray(model_func(x, *theta), dtype=float).ravel()
        J = np.asarray(jacobian_func(x, *theta), dtype=float)
    except Exception:
        return np.full(p, np.nan, dtype=float)

    if J.ndim != 2 or J.shape != (n, p):
        return np.full(p, np.nan, dtype=float)

    residual = y - y_fit
    rss = float(np.sum(residual ** 2))
    dof = n - p
    if dof <= 0:
        return np.full(p, np.nan, dtype=float)

    sigma2 = rss / dof
    JTJ = J.T @ J
    try:
        cov = sigma2 * np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(JTJ)

    diag = np.real(np.diag(cov))
    diag = np.where(np.isfinite(diag) & (diag >= 0), diag, np.nan)
    return np.sqrt(diag)


def safe_eval_equations(eqs_text, symbols):
    """（解释：方程文本解析）
    作用：
        将用户输入的方程文本转为 sympy 表达式列表。
        比如用户输入：
            x**2 + y - 1
            x**2 + y = 1
        这个模块就负责把两个方程转换成可处理的方程组
    """


    #拆分多行方程，去掉空行与空白
    lines = [l.strip() for l in eqs_text.splitlines() if l.strip()]
    #eqs_text.splitlines()把用户输入的乱七八糟的多行公式按照换行符拆分成一块一块的
    #l指代列表中的每一个单元格元素
    #l.strip()表示当前行的字符串

    exprs = []
    #把整理干净的一个个方程挨个放进来


    for line in lines:
        # allow forms like 'x**2 + y - 1' or 'x**2 + y - 1 = 0'
        if '=' in line:
            #如果包含等号，就拆成左右两边
            left, right = line.split('=',1)
            #split表示line列表里，遇到等号就拆开，等号左边给left，等号右边给right，1表示见到一个等号就拆，后面再有等号不管

            expr = sp.sympify(left, locals=symbols) - sp.sympify(right, locals=symbols)
            #转换成左-右=0的形式
            #locals表示大家在这个公式里用到的所有变量名都是前面标记好的，而不是像python语法里那样写一个就是一个新变量，写一个就自定义了一个新的
            #expr代表的是方程中”左边-右边“这部分，不包含=0这部分，所以即便没有写出expr=0，这个表达形式已经说明了expr=0
        else:
            expr = sp.sympify(line, locals=symbols)
            #如果没有等号，说明本身已经是左-右=0的形式了
        exprs.append(expr)
        #转换之后添加到方程列表里
    return exprs

# ============================================================================
#                               主窗口类
# ============================================================================
# -------------------- Main Application --------------------

class MiniMatlabApp(QWidget):
    """Mini MATLAB 主界面类；所有功能都写在此处。"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Mini-MATLAB — 非线性拟合与方程求解Нелинейная аппроксимация и решение уравнений')
        self.resize(1100, 700)

        # 数据/状态存储变量
        self.data = None  # pandas DataFrame
        self.data_path = None
        self.fit_results = None
        self.current_fig = None

        # 建立界面
        self._build_ui()



# ----------------------------- UI 构建 -----------------------------
    def _build_ui(self):
        # 整体左右分栏布局
        main_layout = QHBoxLayout(self) #这是水平布局的函数
        splitter = QSplitter() #允许用户拖拽调整左右布局大小
        main_layout.addWidget(splitter)

        # ===================== 左侧控制区 =====================
        left = QWidget() #创建空白容器
        left_layout = QVBoxLayout(left) #创建垂直布局，并把前面的空白容器放到垂直布局里

        # ---------- 数据导入 ----------
        gb_data = QGroupBox('数据导入/Импорт данных')
        #创建一个带标题的容器框，标题显示为"数据导入"

        gl = QVBoxLayout()
        #创建一个垂直布局管理器

        btn_load_csv = QPushButton('加载 CSV/Загрузить CSV')
        #创建一个按钮，按钮文字为"加载 CSV"

        btn_load_csv.clicked.connect(self.load_csv)
        #将按钮的 点击信号 连接到 槽函数 self.load_csv
        #点击按钮时，自动执行 load_csv() 方法

        btn_load_xlsx = QPushButton('加载 Excel/Загрузить Excel')
        btn_load_xlsx.clicked.connect(self.load_excel)
        #同上

        self.lst_columns = QListWidget()
        #创建一个列表控件，用于显示数据文件的列名

        gl.addWidget(btn_load_csv)
        gl.addWidget(btn_load_xlsx)
        #把小部件btn_load_csv和btn_load_xlsx添加到gl布局里
        #（我明白了，前面只是各种布局的定义，还没有真正开始摆放这些小部件，这里才开始摆放）

        gl.addWidget(QLabel('数据列（选择 x 和 y 列）/Столбцы данных (выберите столбцы x и y)'))
        gl.addWidget(self.lst_columns)
        gb_data.setLayout(gl)
        #将布局 gl 安装 到组框 gb_data上

        left_layout.addWidget(gb_data)
        #将完整的组框添加到父布局 left_layout中



        # ---------- 非线性拟合 ----------
        gb_fit = QGroupBox('非线性拟合/Нелинейная подгонка')
        fit_layout = QFormLayout()

        self.edit_model = QLineEdit('a*exp(-b*x)+c')
        self.edit_params = QLineEdit('a b c')
        self.edit_init = QLineEdit('1 1 0.1')

        # 拟合算法选择：保持原 LM，同时新增 AMLM 备选算法
        self.cmb_algo = QComboBox()
        self.cmb_algo.addItems(['LM (scipy)', 'AMLM (proposed)'])
        #3个文本输入框，设为实例属性（self.）以便在其他方法中读取内容
        #虽然self没有这些属性，但是类的属性是可以现用现定义的，如果前面已经定义过，这里也是可以直接覆盖的
        #QLineEdit是用于单行文本输入的图形控件，相当于一个可编辑的单行文本框，给self添加属性edit_model/edit_params/edit_init，并绑定该文本框
        #这部分也只是做了个定义或者是描述，还没有真正摆放到对应位置去


        self.cmb_xcol = QComboBox()
        self.cmb_ycol = QComboBox()
        #下拉选择框，选择数据列

        btn_fit = QPushButton('开始拟合/Начало установки')
        #一个按钮

        btn_fit.clicked.connect(self.do_fit)
        #给按钮绑定事件do_fit
        
        fit_layout.addRow(QLabel('x 列'), self.cmb_xcol)
        fit_layout.addRow(QLabel('y 列'), self.cmb_ycol)
        fit_layout.addRow(QLabel('模型表达式 (以 x 为自变量)'), self.edit_model)
        fit_layout.addRow(QLabel('参数名（空格分隔）'), self.edit_params)
        fit_layout.addRow(QLabel('初始值（空格分隔）'), self.edit_init)
        fit_layout.addRow(QLabel('拟合算法'), self.cmb_algo)
        fit_layout.addRow(btn_fit)
        #把这些控件们添加到表单布局去


        gb_fit.setLayout(fit_layout)
        #然后把这个表单安装到gb_fit这个组框去

        left_layout.addWidget(gb_fit)
        #把整个组框gb_fit添加到左侧面板主布局中left_layout

        #到这里，左侧布局就变成：
        """
        left_layout (左侧面板)
        ├── gb_data (数据导入组框)
        └── gb_fit  (非线性拟合组框) 
        """

        # ---------- 方程求解 ----------
        gb_eq = QGroupBox('方程求解')
        eq_layout = QFormLayout()
        self.edit_equation = QTextEdit('x**3 - 2*x - 5')
        self.edit_vars = QLineEdit('x')
        self.edit_guess = QLineEdit('2')
        btn_solve = QPushButton('求解方程')
        btn_solve.clicked.connect(self.solve_equation)
        eq_layout.addRow(QLabel('方程（可多行，对应方程组）'), self.edit_equation)
        eq_layout.addRow(QLabel('变量名（空格分隔）'), self.edit_vars)
        eq_layout.addRow(QLabel('初始猜测值（空格分隔）'), self.edit_guess)
        eq_layout.addRow(btn_solve)
        gb_eq.setLayout(eq_layout)
        left_layout.addWidget(gb_eq)
        #同上

        
        
        # ---------- 导出 ----------
        gb_export = QGroupBox('导出 / 保存')
        ex_layout = QVBoxLayout()
        btn_export_params = QPushButton('导出拟合参数 (CSV)')
        btn_export_params.clicked.connect(self.export_params)
        btn_export_plot = QPushButton('导出图像 (PNG)')
        btn_export_plot.clicked.connect(self.export_plot)
        btn_save_proj = QPushButton('保存项目')
        btn_save_proj.clicked.connect(self.save_project)
        btn_load_proj = QPushButton('加载项目')
        btn_load_proj.clicked.connect(self.load_project)
        #这些都是对按钮的定义和事件绑定
        ex_layout.addWidget(btn_export_params)
        ex_layout.addWidget(btn_export_plot)
        ex_layout.addWidget(btn_save_proj)
        ex_layout.addWidget(btn_load_proj)
        gb_export.setLayout(ex_layout)
        left_layout.addWidget(gb_export)
        #这里才是把所有的按钮一个一个排好位置

        left_layout.addStretch()
        #在垂直布局中添加一个弹性空白区域，将上方的控件向上推，使下方剩余空间自动填满



         # ===================== 右侧绘图区 =====================
        right = QWidget()
        #创建右侧面板容器（一个空画框，准备放入图表和日志）
        right_layout = QVBoxLayout(right)
        #在 right 容器内创建垂直布局
        self.fig = Figure(figsize=(5,4))
        #创建 Matplotlib 图表对象，尺寸 5×4 英寸
        self.canvas = FigureCanvas(self.fig)
        #将 Matplotlib 图表 嵌入 到 Qt 界面中（将画布装进画框（使其能在GUI中显示））
        right_layout.addWidget(self.canvas)
        #将图表画布添加到右侧垂直布局（他是第一个放置的，所以在最上边）

        # 清空按钮：清空图像与输出日志
        btn_clear = QPushButton('清空图 / 清空输出')
        btn_clear.clicked.connect(self.clear_view)
        right_layout.addWidget(btn_clear)


        self.txt_log = QTextEdit()
        #创建多行文本框，用于显示程序日志（如拟合过程、错误信息）
        self.txt_log.setReadOnly(True)
        #设置日志框为只读（用户只能看，不能修改）
        right_layout.addWidget(QLabel('日志 / 输出'))
        #在日志框上方添加文字标签，提示该区域功能
        right_layout.addWidget(self.txt_log)
        #将日志文本框添加到垂直布局中

        splitter.addWidget(left)
        #将左侧面板（含数据导入、拟合设置）添加到分割器左侧
        splitter.addWidget(right)
        #将右侧面板（含图表、日志）添加到分割器右侧
        splitter.setSizes([300, 800])

    
    
    # ====================================================================
    #                               数据加载
    # ====================================================================
    def log(self, msg):
        self.txt_log.append(msg) #在界面日志框追加消息
        print(msg)
    #这是一个日志工具方法，用于在 GUI界面 和终端同时输出日志信息


    def clear_view(self):
        """清空图像与日志输出（不影响已加载数据与输入框内容）。"""
        try:
            if self.fig is not None:
                self.fig.clf()
                self.canvas.draw()
        except Exception:
            # 清图失败也不要影响程序继续运行
            pass

        try:
            if self.txt_log is not None:
                self.txt_log.clear()
        except Exception:
            pass

        self.current_fig = None


    def load_csv(self):
    #这是 加载CSV数据文件 的完整方法
    #定义一个实例方法，用于响应"加载CSV"按钮的点击事件
        path, _ = QFileDialog.getOpenFileName(self, '选择 CSV 文件', '', 'CSV Files (*.csv)')
        #弹出文件选择对话框，让用户选择CSV文件
        #path：用户选择的文件路径（如 "/home/data.csv"）
        #_：占位符，接收文件类型（此处不需要，用下划线忽略）
        #self：父窗口（对话框会居中显示）
        #'选择 CSV 文件'：对话框标题
        #''：默认打开目录（空字符串表示当前目录）
        #'CSV Files (*.csv)'：文件过滤器，只显示 .csv 文件

        if not path:
            return
        try:
            df = pd.read_csv(path)
            #读取CSV文件为pandas DataFrame
            #pd.read_csv()：pandas 的核心函数，自动解析CSV格式

        except Exception as e:
            QMessageBox.critical(self, '错误', f'加载 CSV 失败: {e}')
            return
        self._set_data(df, path)
        #调用私有方法 _set_data()，将数据传递给主程序

    def load_excel(self):
        #同上，excel类型文件
        path, _ = QFileDialog.getOpenFileName(self, '选择 Excel 文件', '', 'Excel Files (*.xlsx *.xls)')
        if not path:
            return
        try:
            df = pd.read_excel(path)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'加载 Excel 失败: {e}')
            return
        self._set_data(df, path)

    def _set_data(self, df, path=None):
    #这是一个数据处理与界面同步的核心方法
    #也就是实现：加载-存储-展示-反馈，这么一个闭环
        self.data = df
        self.data_path = path
        self.lst_columns.clear()
        self.cmb_xcol.clear()
        self.cmb_ycol.clear()
        for col in df.columns:
            self.lst_columns.addItem(str(col))
            self.cmb_xcol.addItem(str(col))
            self.cmb_ycol.addItem(str(col))
        self.log(f'已加载数据: {path or "(内存数据)"}，列: {list(df.columns)}')

    # ====================================================================
    #                               拟合
    # ====================================================================
    def do_fit(self):
        if self.data is None: #确保有数据
            QMessageBox.warning(self, '无数据', '请先加载 CSV/Excel 文件。')
            return
        xcol = self.cmb_xcol.currentText() #从x列下拉框读取列名
        ycol = self.cmb_ycol.currentText() #从y列下拉框读取列名
        if not xcol or not ycol: #确保已选择列
            QMessageBox.warning(self, '列未选择', '请选择 x 列和 y 列。')
            return
        x = self.data[xcol].values #从DataFrame中取出x列，转换为NumPy数组
        y = self.data[ycol].values

        model_expr = self.edit_model.text().strip() #从文本框获取用户输入的数学公式，.strip() 去除首尾空格，避免解析错误
        param_names = [p.strip() for p in self.edit_params.text().split() if p.strip()] #解析参数名：将 "a b c" 分割成列表 ['a', 'b', 'c']
        try:
            raw_func = parse_model(model_expr, param_names) #调用之前定义的func函数，1号参数是数学公式，2号参数是公式里的各个参数
        except Exception as e:
            QMessageBox.critical(self, '模型解析失败', str(e))
            return
        # initial values
        try:
            p0 = [float(v) for v in self.edit_init.text().split() if v.strip()]#解析初始值：将 "1 1 0.1" 转换为浮点数列表 [1.0, 1.0, 0.1]
        except Exception as e:
            QMessageBox.critical(self, '初始值错误', f'请确保初始值为数字: {e}')
            return
        if len(p0) != len(param_names): #确保初始值数量与参数名数量一致
            QMessageBox.warning(self, '参数数目不匹配', '初始值数量应与参数名数量一致。')
            return

        algo = self.cmb_algo.currentText() if hasattr(self, 'cmb_algo') else 'LM (scipy)'

        try:
            if algo == 'LM (scipy)':
                # LM：正式跑一次拿结果；runtime 采用 warmup + 多次重复后的最小值，更稳一些
                counted_func = CallCounter(raw_func)

                def run_lm_for_result():
                    return curve_fit(
                        counted_func, x, y, p0=p0, maxfev=20000, full_output=True
                    )

                def run_lm_for_bench():
                    return curve_fit(
                        raw_func, x, y, p0=p0, maxfev=20000, full_output=True
                    )

                popt, pcov, infodict, mesg, ier = run_lm_for_result()
                runtime = benchmark_runtime(run_lm_for_bench, repeat=5, warmup=1)
                perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full_like(popt, np.nan)

                try:
                    y_hat = raw_func(x, *popt)
                    rss = float(np.nansum((np.asarray(y_hat) - np.asarray(y)) ** 2))
                except Exception:
                    rss = float('nan')

                nfev = int(infodict.get('nfev')) if isinstance(infodict, dict) and 'nfev' in infodict else None
                iters = nfev
                ok = bool(ier in (1, 2, 3, 4))

                model_calls = int(counted_func.calls)
                jac_calls = int(estimate_lm_jac_calls(model_calls, len(param_names)))
                eval_cost = int(model_calls + jac_calls)

                self.log(
                    f'使用算法: LM (scipy)，nfev={iters if iters is not None else "?"}，收敛={ok}，RSS={rss:.6g}，runtime={runtime:.6f}s'
                )
                self.log(
                    f'附加指标: model_calls={model_calls}，jac_calls={jac_calls}，eval_cost={eval_cost}'
                )
            else:
                # AMLM：正式跑一次拿结果；runtime 采用 warmup + 多次重复后的最小值，更稳一些
                model_func, jac_func = build_jacobian_sympy(model_expr, param_names)
                counted_model_func = CallCounter(model_func)
                counted_jac_func = CallCounter(jac_func)

                def run_amlm_for_result():
                    return amlm_fit(x, y, counted_model_func, counted_jac_func, p0)

                def run_amlm_for_bench():
                    return amlm_fit(x, y, model_func, jac_func, p0)

                popt, rss, ok, iters, model_calls, jac_calls, eval_cost = run_amlm_for_result()
                runtime = benchmark_runtime(run_amlm_for_bench, repeat=5, warmup=1)
                pcov = None
                # AMLM 之前显示 NaN，是因为这里直接把 perr 人工设成了 NaN。
                # 现在基于最优点处雅可比矩阵的局部线性近似来估计参数标准误差。
                perr = estimate_parameter_std_errors(x, y, popt, model_func, jac_func)
                self.log(f'使用算法: AMLM (proposed)，迭代 {iters} 次，收敛={ok}，RSS={rss:.6g}，runtime={runtime:.6f}s')
                self.log(
                    f'附加指标: model_calls={model_calls}，jac_calls={jac_calls}，eval_cost={eval_cost}'
                )
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, '拟合失败', f'拟合过程中出现错误: {e}\n{tb}')
            return

        #计算参数标准误差：如果拟合成功，从协方差矩阵对角线计算误差
        #如果pcov不为空，那么提取该矩阵对角线元素(diag)，然后计算开方

        self.fit_results = {
            'model': model_expr,
            'algo': algo,
            'param_names': param_names,
            'popt': popt.tolist(),
            'perr': perr.tolist(),
            'xcol': xcol,
            'ycol': ycol,
            'rss': rss,
            'iters': iters,
            'ok': ok,
            'model_calls': model_calls,
            'jac_calls': jac_calls,
            'eval_cost': eval_cost,
            'runtime': runtime
        }
        #存储结果：将拟合结果打包成字典，保存为实例属性

        self.log(f'拟合完成。参数: {list(zip(param_names, popt, perr))}')
        self.log(
            f'结果汇总: RSS={rss:.6g}，收敛={ok}，迭代次数/nfev={iters}，runtime={runtime:.6f}s，model_calls={model_calls}，jac_calls={jac_calls}，eval_cost={eval_cost}'
        )
        #日志输出：在界面和终端显示拟合结果

        self._plot_fit(x, y, raw_func, popt)
        #绘图更新：调用自定义的 _plot_fit 方法

    def _plot_fit(self, x, y, func, popt):
        #x, y: 原始数据数组（横纵坐标）
        #func: 拟合函数（已编译的可调用对象）
        #popt: 最优参数列表（如 [2.5, 0.8, 0.5]）
        self.fig.clf()
        #清空整个图表，防止多次拟合后图像重叠（每次绘图前清零）
        ax = self.fig.add_subplot(111)
        #在 self.fig 中创建一个单个子图并返回坐标轴对象 ax
        #后续所有绘图操作都通过 ax. 调用

        ax.scatter(x, y, label='data')
        #绘制散点图：将原始数据点绘制为散点

        xs = np.linspace(np.nanmin(x), np.nanmax(x), 500)
        #生成光滑曲线所需x值：
        #np.nanmin(x): x的最小值（忽略NaN）
        #np.nanmax(x): x的最大值（忽略NaN）
        #示例：如果 x 范围是 [0, 10]，则 xs 为 [0, 0.02, 0.04, ..., 10]

        try:
            ys = func(xs, *popt)
            #*popt：解包参数列表，等价于 func(xs, popt[0], popt[1], popt[2])
        except Exception as e:
            ys = np.zeros_like(xs)
            self.log(f'绘图评估模型失败: {e}')

        ax.plot(xs, ys, label='fit', linewidth=2)
        #绘制拟合曲线：用蓝色实线（默认）连接500个点，形成光滑曲线
        ax.set_xlabel(self.fit_results.get('xcol','x'))
        #设置x轴标签
        ax.set_ylabel(self.fit_results.get('ycol','y'))
        #设置y轴标签
        ax.legend()
        self.canvas.draw()
        self.current_fig = self.fig
        #将当前图表对象保存到 self.current_fig

    # ====================================================================
    #                               求解方程
    # ====================================================================
    def solve_equation(self):
        #数值求解方程组（用计算机求解，而非解析公式）
        eq_text = self.edit_equation.toPlainText().strip() #输入的数学公式
        if not eq_text:
            QMessageBox.warning(self, '输入为空', '请输入要解的方程或方程组。')
            return
        var_names = [v.strip() for v in self.edit_vars.text().split() if v.strip()]
        #self.edit_vars.text().split()获取文本编辑框中的原始字符串，按空白字符分割字符串，返回词元列表，例："a b c" → ['a', 'b', 'c']
        #v拿到每个词元，然后strip去除空白
        if not var_names:
            QMessageBox.warning(self, '变量未填写', '请填写变量名（空格分隔）。')
            return
        guess_vals = [float(v) for v in self.edit_guess.text().split() if v.strip()]
        #把文本编辑框中的字符转化为浮点型
        if len(guess_vals) != len(var_names):
            QMessageBox.warning(self, '猜测值数目不匹配', '初始猜测值数量应与变量数量一致。')
            return
        # create sympy symbols
        syms = {name: sp.symbols(name) for name in var_names}
        #var_names得到的每个变量给name，使用symbols命令生成对应的符号变量，并和name组成键值对
        #例如{'x': x, 'y': y, 'z': z}
        try:
            exprs = safe_eval_equations(eq_text, syms)
            #将用户输入的文本方程转换为 SymPy 表达式对象
            #eq_text字符串，包含用户输入的方程/表达式
            #syms符号字典
            #例如：用户输入eq_text =  x**2 + y**2 = 1，x - y = 0.5
            #输出：[-x**2 - y**2 + 1, -x + y + 0.5]

        except Exception as e:
            QMessageBox.critical(self, '解析方程失败', str(e))
            return
        # convert to numeric function for root-finding
        try:
            f_lambdas = [sp.lambdify(tuple(syms.values()), e, modules=['numpy']) if not isinstance(e, sp.Equality) else sp.lambdify(tuple(syms.values()), e.lhs - e.rhs, modules=['numpy']) for e in exprs]
            # 前面把用户输入转化成了符号表达式，这里再转化为了数值函数
            # 其实符号表达式已经可以用于计算机处理了，但是符号表达式每次都要重建符号树、类型检查、符号匹配，会产生巨大的 Python 解释器开销
            # 而数值函数直接编译为机器码，向量化批量计算，最小化解释器调用
            # 所以其实转化为数值函数就是为了后面的大规模计算
            # 最后的返回结果就是用户输入的多个表达式
        except Exception as e:
            QMessageBox.critical(self, '转为数值函数失败', str(e))
            return

        def fun(vals):
            #这个函数的目的是求，在当前参数下，方程的残差
            #“残差”和误差的区别在于，误差需要知道精确值，精确值-估计值=误差
            #残差就是不知道精确值时，例如：
            #x-3=0，估计x=2.9，带入公式，得到残差为-0.1，再估计x=3.01，带入公式，残差为0.01，说明x2更接近真实解
            #然后再再估计x=3，带入公式，残差为0，说明3就是真实解
            vals = np.array(vals)
            return [float(f(*vals)) for f in f_lambdas]

        try:
            sol = root(fun, guess_vals)
            #调用 SciPy 的 root() 函数 对非线性方程组进行数值求解
            #root()的底层逻辑代码可以自己去搜一下，这里只是调用了python自带的求解方法
        except Exception as e:
            QMessageBox.critical(self, '求解失败', f'求解过程报错: {e}')
            return
        if not sol.success:
            QMessageBox.warning(self, '求解可能失败', f'求解器未收敛: {sol.message}')
        res = {name: float(val) for name, val in zip(var_names, sol.x)}
        self.log(f'方程求解结果: {res}')
        QMessageBox.information(self, '解', json.dumps(res, ensure_ascii=False, indent=2))

    # ====================================================================
    #                               导出/保存
    # ====================================================================
    def export_params(self):
        if not self.fit_results:
            QMessageBox.warning(self, '无拟合结果', '请先进行拟合。')
            return
        path, _ = QFileDialog.getSaveFileName(self, '保存拟合参数', 'fit_params.csv', 'CSV 文件 (*.csv);;Excel 文件 (*.xlsx)')
        if not path:
            return
        df = pd.DataFrame({
            'param': self.fit_results['param_names'],
            'value': self.fit_results['popt'],
            'stderr': self.fit_results['perr']
        })
        try:
            if path.lower().endswith('.xlsx'):
                df.to_excel(path, index=False)
            else:
                df.to_csv(path, index=False)
            self.log(f'已导出拟合参数: {path}')
        except Exception as e:
            QMessageBox.critical(self, '导出失败', str(e))

    def export_plot(self):
        if self.current_fig is None:
            QMessageBox.warning(self, '无图像', '请先进行绘图（例如拟合）。')
            return
        path, _ = QFileDialog.getSaveFileName(self, '保存图像', 'figure.png', 'PNG 图片 (*.png);;SVG 矢量图 (*.svg)')
        if not path:
            return
        try:
            self.current_fig.savefig(path)
            self.log(f'已保存图像: {path}')
        except Exception as e:
            QMessageBox.critical(self, '保存失败', str(e))

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, '保存项目 (JSON)', 'project.json', 'JSON 文件 (*.json)')
        if not path:
            return
        proj = {
            'data_path': self.data_path,
            'model': self.fit_results or None,
            'model_expr': self.edit_model.text(),
            'model_params': self.edit_params.text(),
            'init': self.edit_init.text(),
            'equation': self.edit_equation.toPlainText(),
            'vars': self.edit_vars.text(),
            'guess': self.edit_guess.text()
        }
        try:
            with open(path, 'w', encoding='utf8') as f:
                json.dump(proj, f, ensure_ascii=False, indent=2)
            self.log(f'已保存项目: {path}')
        except Exception as e:
            QMessageBox.critical(self, '保存失败', str(e))

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, '加载项目 (JSON)', '', 'JSON 文件 (*.json)')
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf8') as f:
                proj = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, '加载失败', str(e))
            return
        # restore
        if proj.get('data_path') and os.path.exists(proj['data_path']):
            try:
                df = pd.read_csv(proj['data_path']) if proj['data_path'].lower().endswith('.csv') else pd.read_excel(proj['data_path'])
                self._set_data(df, proj['data_path'])
            except Exception as e:
                self.log(f'加载项目中数据文件失败: {e}')
        self.edit_model.setText(proj.get('model_expr',''))
        self.edit_params.setText(proj.get('model_params',''))
        self.edit_init.setText(proj.get('init',''))
        self.edit_equation.setPlainText(proj.get('equation',''))
        self.edit_vars.setText(proj.get('vars',''))
        self.edit_guess.setText(proj.get('guess',''))
        self.log(f'项目已加载: {path}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    #设置全局字体
    from PyQt5 import QtGui
    font = QtGui.QFont()
    font.setPointSize(12)
    app.setFont(font)


    win = MiniMatlabApp()
    win.show()
    sys.exit(app.exec_())
