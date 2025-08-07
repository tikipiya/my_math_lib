#my_math_lib_main_ver0.01.py
import math
import random

#定数定義
class const:
    #数学定数
    pi = 3.1415926535897932384                  # 円周率（20桁）
    e = 2.7182818284590452353                   # ネイピア数（20桁）
    phi = 0.6180339887498948482                 # 黄金数（20桁）
    apery = 1.2020569031595942853                # アペリーの定数（20桁）
    gelf = 1.6325269194381528447                 # ゲルフォント・シュナイダー定数（20桁）
    pla = 1.3247179572447460259                  # プラスチック数（19桁 → 真値）
    euler = 0.5772156649015328606                # オイラー定数（20桁）
    catalan = 0.91596559417721901505             # カタラン定数（20桁）
    khinchin = 2.68545200106530644530            # キンチン定数（20桁）
    feigenbaum_d = 4.66920160910299067185        # Feigenbaum定数 δ（20桁）
    feigenbaum_a = 2.50290787509589282228        # Feigenbaum定数 α（20桁）
    glaisher = 1.28242712910062263687             # Glaisher-Kinkelin定数
    euler_gamma = 0.57721566490153286060          # オイラー・マスケローニ定数（重複注意）
    meissel_mertens = 0.26149721284764278375      # Meissel–Mertens定数
    twin_prime_const = 0.66016181584686957392     # 双子素数定数
    brass = 1.45607494858268967145                 # ブラス定数
    yang_lee_edge = 0.29559774252203960404         # Yang–Leeのエッジ定数
    gauss_kuzmin = 0.30366300289873263056          # Gauss-Kuzmin定数
    vilars_constant = 0.76422365358922066299       # Vilarsの定数

    #物理定数（2024 CODATA）
    c = 299792458                                 # 光速 [m/s]（定義値）
    G = 6.67430e-11                               # 万有引力定数 [m^3 kg^-1 s^-2]
    h = 6.62607015e-34                            # プランク定数 [J·s]
    hbar = 1.054571817e-34                         # ディラック定数 ħ = h / (2π)
    k = 1.380649e-23                              # ボルツマン定数 [J/K]
    NA = 6.02214076e23                            # アボガドロ定数 [/mol]
    qe = 1.602176634e-19                          # 素電荷 [C]
    eps0 = 8.8541878128e-12                       # 真空の誘電率 [F/m]
    mu0 = 1.25663706212e-6                        # 真空の透磁率 [N/A²]
    R = 8.314462618                               # 気体定数 [J/mol·K]
    alpha = 0.0072973525693                        # 微細構造定数（無次元）
    Rydberg = 10973731.568160                      # リュードベリ定数 [m^-1]
    mu_B = 9.274009994e-24                         # ボーア磁子 [J/T]
    mu_N = 5.0507837461e-27                        # 核磁子 [J/T]
    eV = 1.602176634e-19                           # 電子ボルト（素電荷と同じ）
    fine_structure = 7.2973525693e-3               # 微細構造定数（無次元、別名）
    hartree_energy = 4.3597447222071e-18           # ハートリーエネルギー [J]
    rydberg_energy = 2.1798723611035e-18            # リュードベリエネルギー [J]
    planck_length = 1.616255e-35                    # プランク長 [m]
    planck_time = 5.391247e-44                       # プランク時間 [s]
    stefan_boltzmann = 5.670374419e-8               # ステファン・ボルツマン定数 [W m^-2 K^-4]
    gas_constant = 8.314462618                       # 気体定数（Rと同値）
    bohr_radius = 5.29177210903e-11                  # ボーア半径 [m]

    #化学定数
    atm = 101325.0                                  # 標準大気圧 [Pa]
    molar_gas_const = 8.314462618                   # 気体定数（Rと同値）
    Faraday = 96485.33212                           # ファラデー定数 [C/mol]
    standard_pressure = 101325                       # 標準圧力 [Pa]
    avogadro = 6.02214076e23                         # アボガドロ定数（NAと同じ）
    boltzmann = 1.380649e-23                         # ボルツマン定数（kと同じ）
# クラス外に全部展開
pi = const.pi
e = const.e
phi = const.phi
apery = const.apery
gelf = const.gelf
pla = const.pla
euler = const.euler
catalan = const.catalan
khinchin = const.khinchin
feigenbaum_d = const.feigenbaum_d
feigenbaum_a = const.feigenbaum_a
glaisher = const.glaisher
euler_gamma = const.euler_gamma
meissel_mertens = const.meissel_mertens
twin_prime_const = const.twin_prime_const
brass = const.brass
yang_lee_edge = const.yang_lee_edge
gauss_kuzmin = const.gauss_kuzmin
vilars_constant = const.vilars_constant
c = const.c
G = const.G
h = const.h
hbar = const.hbar
k = const.k
NA = const.NA
qe = const.qe
eps0 = const.eps0
mu0 = const.mu0
R = const.R
alpha = const.alpha
Rydberg = const.Rydberg
mu_B = const.mu_B
mu_N = const.mu_N
eV = const.eV
fine_structure = const.fine_structure
hartree_energy = const.hartree_energy
rydberg_energy = const.rydberg_energy
planck_length = const.planck_length
planck_time = const.planck_time
stefan_boltzmann = const.stefan_boltzmann
gas_constant = const.gas_constant
bohr_radius = const.bohr_radius
atm = const.atm
molar_gas_const = const.molar_gas_const
Faraday = const.Faraday
standard_pressure = const.standard_pressure
avogadro = const.avogadro
boltzmann = const.boltzmann



def sin(x):
    # xはラジアン
    # 10項までのテイラー展開
    s = 0
    num = x
    sign = 1
    fact = 1
    for i in range(1, 20, 2):
        s += sign * num / fact
        num *= x * x
        fact *= (i + 1) * (i + 2)
        sign *= -1
    return s

def sqrt(x):
    # ニュートン法
    if x < 0:
        raise ValueError("sqrt of negative number")
    if x == 0:
        return 0
    guess = x
    for _ in range(20):
        guess = (guess + x / guess) / 2
    return guess

def cos(x):
    # x in radians, 10 terms Taylor expansion
    s = 0
    num = 1.0
    sign = 1
    fact = 1
    for i in range(0, 20, 2):
        s += sign * num / fact
        num *= x * x
        fact *= (i + 1) * (i + 2)
        sign *= -1
    return s

def tan(x):
    c = cos(x)
    if abs(c) < 1e-12:
        raise ValueError("tan(x): cos(x) is too close to zero")
    return sin(x) / c

def ln(x):
    """高精度自然対数（ビルトインのみ, 20桁目標）"""
    if x <= 0:
        raise ValueError("ln(x) is undefined for x <= 0")
    # x = m * 2^k に変形
    k = 0
    while x > 2:
        x /= 2
        k += 1
    while x < 1:
        x *= 2
        k -= 1
    # ln(x) = k*ln(2) + ln(y) (1 <= y < 2)
    # ln(2)をメリカトリ級数で高精度に求める
    def ln2_mer():
        t = 1  # ln(2) = log(1+1)
        s = 0
        sign = 1
        for n in range(1, 100):
            term = sign * (t**n) / n
            s += term
            sign *= -1
            if abs(term) < 1e-21:
                break
        return s
    ln2 = ln2_mer()
    # x = 1 + t (0 < t < 1)
    t = x - 1
    s = 0
    sign = 1
    for n in range(1, 200):
        term = sign * (t**n) / n
        s += term
        sign *= -1
        if abs(term) < 1e-21:
            break
    return k * ln2 + s

def fact(n):
    res = 1
    for i in range(2, int(n)+1):
        res *= i
    return res

def C(n, r):
    n = int(n)
    r = int(r)
    if r < 0 or n < 0 or r > n:
        return 0
    return fact(n) // (fact(r) * fact(n - r))

def P(n, r):
    n = int(n)
    r = int(r)
    if r < 0 or n < 0 or r > n:
        return 0
    return fact(n) // fact(n - r)

def mean(lst):return sum(lst) / len(lst) if len(lst) > 0 else 0

def deviation(lst):
    m = mean(lst)
    return [x - m for x in lst]

def stddev(lst):
    m = mean(lst)
    if len(lst) < 2:
        return 0
    return (sum((x - m)**2 for x in lst) / (len(lst)-1)) ** 0.5

def sumall(lst):return sum(lst)

def prodall(lst):
    res = 1
    for x in lst:
        res *= x
    return res

# 逆三角関数
def arcsin(x):
    """逆正弦関数（-1 <= x <= 1）"""
    if x < -1 or x > 1:
        raise ValueError("arcsin(x) is undefined for |x| > 1")
    if x == 1:
        return pi / 2
    if x == -1:
        return -pi / 2
    if x == 0:
        return 0
    
    # ニュートン法で arcsin(x) を求める
    # sin(y) = x となる y を求める
    y = x  # 初期値
    for _ in range(20):
        sin_y = sin(y)
        cos_y = cos(y)
        if abs(cos_y) < 1e-12:
            break
        y = y - (sin_y - x) / cos_y
    return y

def arccos(x):
    """逆余弦関数（-1 <= x <= 1）"""
    if x < -1 or x > 1:
        raise ValueError("arccos(x) is undefined for |x| > 1")
    return pi / 2 - arcsin(x)

def arctan(x):
    """逆正接関数"""
    if x == 0:
        return 0
    if x == 1:
        return pi / 4
    if x == -1:
        return -pi / 4
    
    # ニュートン法で arctan(x) を求める
    # tan(y) = x となる y を求める
    y = x  # 初期値
    for _ in range(20):
        tan_y = tan(y)
        sec2_y = 1 / (cos(y) ** 2)
        y = y - (tan_y - x) / sec2_y
    return y

# 双曲線関数
def sinh(x):
    """双曲線正弦関数"""
    return (exp(x) - exp(-x)) / 2

def cosh(x):
    """双曲線余弦関数"""
    return (exp(x) + exp(-x)) / 2

def tanh(x):
    """双曲線正接関数"""
    ch = cosh(x)
    if abs(ch) < 1e-12:
        raise ValueError("tanh(x): cosh(x) is too close to zero")
    return sinh(x) / ch

# 指数関数
def exp(x):
    """指数関数 e^x"""
    # テイラー展開
    s = 1.0
    term = 1.0
    for i in range(1, 50):
        term *= x / i
        s += term
        if abs(term) < 1e-15:
            break
    return s

# べき乗関数
def pow(x, y):
    """x^y"""
    if x == 0 and y <= 0:
        raise ValueError("0^y is undefined for y <= 0")
    if x < 0 and y != int(y):
        raise ValueError("x^y is undefined for x < 0 and y not integer")
    
    if y == 0:
        return 1
    if y == 1:
        return x
    if y == 2:
        return x * x
    if y == -1:
        return 1 / x
    
    # y = n + f (n: 整数部分, f: 小数部分)
    n = int(y)
    f = y - n
    
    if f == 0:
        # 整数べき乗
        if n > 0:
            result = 1
            for _ in range(n):
                result *= x
            return result
        else:
            result = 1
            for _ in range(-n):
                result *= x
            return 1 / result
    else:
        # 実数べき乗: x^y = x^n * x^f = x^n * e^(f * ln(x))
        if x < 0:
            raise ValueError("x^y is undefined for x < 0 and y not integer")
        return pow(x, n) * exp(f * ln(x))

# 統計関数の拡張
def variance(lst):
    """分散"""
    m = mean(lst)
    if len(lst) < 2:
        return 0
    return sum((x - m) ** 2 for x in lst) / (len(lst) - 1)

def median(lst):
    """中央値"""
    if len(lst) == 0:
        return 0
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    if n % 2 == 0:
        return (sorted_lst[n//2 - 1] + sorted_lst[n//2]) / 2
    else:
        return sorted_lst[n//2]

def mode(lst):
    """最頻値"""
    if len(lst) == 0:
        return 0
    counts = {}
    for x in lst:
        counts[x] = counts.get(x, 0) + 1
    max_count = max(counts.values())
    modes = [x for x, count in counts.items() if count == max_count]
    return modes[0] if len(modes) == 1 else modes

def minmax(lst):
    """最小値と最大値のタプルを返す"""
    if len(lst) == 0:
        return (0, 0)
    return (min(lst), max(lst))

def range_val(lst):
    """範囲（最大値 - 最小値）"""
    if len(lst) == 0:
        return 0
    return max(lst) - min(lst)

# 行列演算
class Matrix:
    def __init__(self, data):
        """行列の初期化"""
        if not data or not data[0]:
            raise ValueError("Matrix cannot be empty")
        self.rows = len(data)
        self.cols = len(data[0])
        self.data = data
        
        # 全ての行が同じ長さかチェック
        for row in data:
            if len(row) != self.cols:
                raise ValueError("All rows must have the same length")
    
    def __getitem__(self, key):
        """行列要素へのアクセス"""
        i, j = key
        return self.data[i][j]
    
    def __setitem__(self, key, value):
        """行列要素の設定"""
        i, j = key
        self.data[i][j] = value
    
    def __str__(self):
        """文字列表現"""
        return '\n'.join([' '.join(map(str, row)) for row in self.data])
    
    def __add__(self, other):
        """行列の加算"""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition")
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self[i, j] + other[i, j])
            result.append(row)
        return Matrix(result)
    
    def __mul__(self, other):
        """行列の乗算"""
        if isinstance(other, (int, float)):
            # スカラー乗算
            result = []
            for i in range(self.rows):
                row = []
                for j in range(self.cols):
                    row.append(self[i, j] * other)
                result.append(row)
            return Matrix(result)
        else:
            # 行列乗算
            if self.cols != other.rows:
                raise ValueError("Matrix dimensions must be compatible for multiplication")
            result = []
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    sum_val = 0
                    for k in range(self.cols):
                        sum_val += self[i, k] * other[k, j]
                    row.append(sum_val)
                result.append(row)
            return Matrix(result)
    
    def transpose(self):
        """転置行列"""
        result = []
        for j in range(self.cols):
            row = []
            for i in range(self.rows):
                row.append(self[i, j])
            result.append(row)
        return Matrix(result)
    
    def det(self):
        """行列式（2x2行列のみ）"""
        if self.rows != self.cols:
            raise ValueError("Determinant is only defined for square matrices")
        if self.rows == 2:
            return self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]
        else:
            raise NotImplementedError("Determinant for matrices larger than 2x2 not implemented")

def create_matrix(rows, cols, value=0):
    """指定サイズの行列を作成"""
    return Matrix([[value for _ in range(cols)] for _ in range(rows)])

def identity_matrix(n):
    """n次単位行列"""
    data = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(1 if i == j else 0)
        data.append(row)
    return Matrix(data)

# 複素数演算
class Complex:
    def __init__(self, real, imag=0):
        """複素数の初期化"""
        self.real = real
        self.imag = imag
    
    def __str__(self):
        """文字列表現"""
        if self.imag >= 0:
            return f"{self.real} + {self.imag}i"
        else:
            return f"{self.real} - {abs(self.imag)}i"
    
    def __add__(self, other):
        """複素数の加算"""
        if isinstance(other, (int, float)):
            return Complex(self.real + other, self.imag)
        return Complex(self.real + other.real, self.imag + other.imag)
    
    def __mul__(self, other):
        """複素数の乗算"""
        if isinstance(other, (int, float)):
            return Complex(self.real * other, self.imag * other)
        return Complex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )
    
    def __abs__(self):
        """複素数の絶対値"""
        return sqrt(self.real * self.real + self.imag * self.imag)
    
    def conjugate(self):
        """共役複素数"""
        return Complex(self.real, -self.imag)

def complex_sqrt(x):
    """複素数の平方根"""
    if isinstance(x, (int, float)):
        if x >= 0:
            return sqrt(x)
        else:
            return Complex(0, sqrt(-x))
    else:
        # 複素数の平方根
        r = abs(x)
        theta = arctan(x.imag / x.real) if x.real != 0 else pi/2
        sqrt_r = sqrt(r)
        return Complex(sqrt_r * cos(theta/2), sqrt_r * sin(theta/2))

# 数値積分
def integrate(f, a, b, n=1000):
    """数値積分（台形公式）"""
    if a >= b:
        raise ValueError("Integration bounds must be a < b")
    
    h = (b - a) / n
    sum_val = (f(a) + f(b)) / 2
    
    for i in range(1, n):
        x = a + i * h
        sum_val += f(x)
    
    return h * sum_val

def integrate_simpson(f, a, b, n=1000):
    """数値積分（シンプソン公式）"""
    if a >= b:
        raise ValueError("Integration bounds must be a < b")
    if n % 2 != 0:
        n += 1  # n must be even
    
    h = (b - a) / n
    sum_val = f(a) + f(b)
    
    for i in range(1, n, 2):
        x = a + i * h
        sum_val += 4 * f(x)
    
    for i in range(2, n-1, 2):
        x = a + i * h
        sum_val += 2 * f(x)
    
    return h * sum_val / 3

# 微分
def derivative(f, x, h=1e-8):
    """数値微分"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 方程式の解
def newton_method(f, df, x0, tol=1e-10, max_iter=100):
    """ニュートン法で方程式 f(x) = 0 の解を求める"""
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = df(x)
        if abs(dfx) < 1e-12:
            raise ValueError("Derivative too close to zero")
        x = x - fx / dfx
    raise ValueError("Newton method did not converge")

# ベクトル演算
def dot_product(v1, v2):
    """ベクトルの内積"""
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    return sum(a * b for a, b in zip(v1, v2))

def cross_product(v1, v2):
    """ベクトルの外積（3次元のみ）"""
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Cross product is only defined for 3D vectors")
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ]

def vector_norm(v):
    """ベクトルのノルム"""
    return sqrt(sum(x * x for x in v))

# 確率・統計
def factorial(n):
    """階乗（fact関数の別名）"""
    return fact(n)

def binomial_probability(n, k, p):
    """二項確率 P(X = k) = C(n,k) * p^k * (1-p)^(n-k)"""
    if k < 0 or k > n or p < 0 or p > 1:
        return 0
    return C(n, k) * pow(p, k) * pow(1 - p, n - k)

def poisson_probability(lambda_val, k):
    """ポアソン確率 P(X = k) = λ^k * e^(-λ) / k!"""
    if lambda_val < 0 or k < 0:
        return 0
    return pow(lambda_val, k) * exp(-lambda_val) / fact(k)

# 幾何学関数
def distance_2d(p1, p2):
    """2次元平面上の2点間の距離"""
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def distance_3d(p1, p2):
    """3次元空間での2点間の距離"""
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def area_circle(r):
    """円の面積"""
    return pi * r * r

def circumference_circle(r):
    """円の円周"""
    return 2 * pi * r

def volume_sphere(r):
    """球の体積"""
    return 4/3 * pi * r * r * r

def surface_area_sphere(r):
    """球の表面積"""
    return 4 * pi * r * r

# ユーティリティ関数
def is_prime(n):
    """素数判定"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def gcd(a, b):
    """最大公約数（ユークリッドの互除法）"""
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """最小公倍数"""
    return abs(a * b) // gcd(a, b)

def fibonacci(n):
    """フィボナッチ数列の第n項"""
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def fibonacci_list(n):
    """フィボナッチ数列の最初のn項"""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

# 角度変換
def deg_to_rad(degrees):
    """度からラジアンへの変換"""
    return degrees * pi / 180

def rad_to_deg(radians):
    """ラジアンから度への変換"""
    return radians * 180 / pi

# 対数関数の拡張
def log(x, base=10):
    """底がbaseの対数"""
    if x <= 0 or base <= 0 or base == 1:
        raise ValueError("Invalid arguments for logarithm")
    return ln(x) / ln(base)

def log10(x):
    """常用対数"""
    return log(x, 10)

def log2(x):
    """底が2の対数"""
    return log(x, 2)

# 化学分野の関数
def molarity_to_molality(molarity, density, molar_mass):
    """モル濃度からモル濃度への変換"""
    # molality = molarity / (density - molarity * molar_mass / 1000)
    return molarity / (density - molarity * molar_mass / 1000)

def molality_to_molarity(molality, density, molar_mass):
    """モル濃度からモル濃度への変換"""
    # molarity = molality * density / (1 + molality * molar_mass / 1000)
    return molality * density / (1 + molality * molar_mass / 1000)

def ph_to_h_concentration(ph):
    """pHから水素イオン濃度への変換"""
    return pow(10, -ph)

def h_concentration_to_ph(h_concentration):
    """水素イオン濃度からpHへの変換"""
    return -log10(h_concentration)

def pka_to_ka(pka):
    """pKaからKaへの変換"""
    return pow(10, -pka)

def ka_to_pka(ka):
    """KaからpKaへの変換"""
    return -log10(ka)

def buffer_ph(pka, acid_concentration, base_concentration):
    """ヘンダーソン・ハッセルバルヒの式によるpH計算"""
    return pka + log10(base_concentration / acid_concentration)

def equilibrium_constant_kp(kc, delta_n, temperature):
    """KcからKpへの変換"""
    # Kp = Kc * (RT)^(Δn)
    return kc * pow(R * temperature, delta_n)

def van_der_waals_pressure(n, v, t, a, b):
    """ファンデルワールス方程式による圧力計算"""
    # P = nRT/(V-nb) - a(n/V)^2
    return (n * R * t) / (v - n * b) - a * pow(n / v, 2)

def ideal_gas_law_pressure(n, v, t):
    """理想気体の法則による圧力計算"""
    return n * R * t / v

def ideal_gas_law_volume(n, p, t):
    """理想気体の法則による体積計算"""
    return n * R * t / p

def ideal_gas_law_temperature(p, v, n):
    """理想気体の法則による温度計算"""
    return p * v / (n * R)

def arrhenius_equation(k, a, ea, t):
    """アレニウスの式による反応速度定数計算"""
    # k = A * exp(-Ea/(RT))
    return a * exp(-ea / (R * t))

def arrhenius_activation_energy(k1, k2, t1, t2):
    """アレニウスの式から活性化エネルギーを計算"""
    # Ea = R * ln(k2/k1) / (1/T1 - 1/T2)
    return R * ln(k2 / k1) / (1/t1 - 1/t2)

def colligative_property_boiling_point_elevation(kb, molality):
    """沸点上昇度の計算"""
    return kb * molality

def colligative_property_freezing_point_depression(kf, molality):
    """凝固点降下度の計算"""
    return kf * molality

def osmotic_pressure(molarity, temperature):
    """浸透圧の計算"""
    return molarity * R * temperature

def beer_lambert_absorbance(epsilon, concentration, path_length):
    """ベール・ランベルトの法則による吸光度計算"""
    return epsilon * concentration * path_length

def beer_lambert_concentration(absorbance, epsilon, path_length):
    """ベール・ランベルトの法則による濃度計算"""
    return absorbance / (epsilon * path_length)

def nernst_equation(e0, temperature, n, q_reaction, q_products):
    """ネルンスト方程式による電極電位計算"""
    return e0 - (R * temperature / (n * Faraday)) * ln(q_products / q_reaction)

def gibbs_free_energy_change(delta_h, delta_s, temperature):
    """ギブズ自由エネルギー変化の計算"""
    return delta_h - temperature * delta_s

def entropy_change_ideal_gas(n, v2, v1):
    """理想気体のエントロピー変化"""
    return n * R * ln(v2 / v1)

def enthalpy_change_constant_pressure(cp, delta_t):
    """定圧熱容量によるエンタルピー変化"""
    return cp * delta_t

# 物理分野の関数
def kinetic_energy(mass, velocity):
    """運動エネルギー"""
    return 0.5 * mass * velocity * velocity

def potential_energy_gravity(mass, height, g=9.81):
    """重力による位置エネルギー"""
    return mass * g * height

def gravitational_force(m1, m2, r):
    """万有引力"""
    return G * m1 * m2 / (r * r)

def coulomb_force(q1, q2, r):
    """クーロン力"""
    return (1 / (4 * pi * eps0)) * q1 * q2 / (r * r)

def electric_field_charge(q, r):
    """点電荷による電場"""
    return (1 / (4 * pi * eps0)) * q / (r * r)

def magnetic_field_wire(current, distance):
    """直線電流による磁場"""
    return (mu0 / (2 * pi)) * current / distance

def lorentz_force(q, e, v, b):
    """ローレンツ力（スカラー版）"""
    # F = q(E + v × B)
    return q * (e + v * b)

def doppler_shift(f0, v_source, v_observer, c_sound=343):
    """ドップラー効果による周波数変化"""
    return f0 * (c_sound + v_observer) / (c_sound - v_source)

def relativistic_mass(m0, velocity):
    """相対論的質量"""
    gamma = 1 / sqrt(1 - (velocity * velocity) / (c * c))
    return m0 * gamma

def relativistic_energy(mass):
    """相対論的エネルギー"""
    return mass * c * c

def de_broglie_wavelength(momentum):
    """ド・ブロイ波長"""
    return h / momentum

def heisenberg_uncertainty_position_momentum(delta_x):
    """ハイゼンベルクの不確定性原理（位置・運動量）"""
    return hbar / (2 * delta_x)

def heisenberg_uncertainty_energy_time(delta_t):
    """ハイゼンベルクの不確定性原理（エネルギー・時間）"""
    return hbar / (2 * delta_t)

def bohr_radius_energy(n):
    """ボーア半径でのエネルギー"""
    return -13.6 / (n * n)  # eV

def rydberg_energy_transition(n1, n2):
    """リュードベリエネルギー遷移"""
    return Rydberg * (1/(n1*n1) - 1/(n2*n2))

def planck_energy(frequency):
    """プランクエネルギー"""
    return h * frequency

def blackbody_radiation_intensity(temperature, wavelength):
    """黒体放射強度（ウィーンの法則近似）"""
    # I(λ,T) = (2πhc²/λ⁵) * 1/(e^(hc/λkT) - 1)
    hc_lambda_kt = (h * c) / (wavelength * k * temperature)
    return (2 * pi * h * c * c) / pow(wavelength, 5) / (exp(hc_lambda_kt) - 1)

def wien_displacement_law(temperature):
    """ウィーンの変位則"""
    return 2.8977729e-3 / temperature  # m⋅K

def stefan_boltzmann_power(temperature, area):
    """ステファン・ボルツマンの法則による放射パワー"""
    return stefan_boltzmann * area * pow(temperature, 4)

def fluid_pressure(density, height, g=9.81):
    """流体圧力"""
    return density * g * height

def bernoulli_equation(p1, rho1, v1, h1, p2, rho2, v2, h2, g=9.81):
    """ベルヌーイの式"""
    # P₁ + ½ρ₁v₁² + ρ₁gh₁ = P₂ + ½ρ₂v₂² + ρ₂gh₂
    return p1 + 0.5 * rho1 * v1 * v1 + rho1 * g * h1

def reynolds_number(density, velocity, diameter, viscosity):
    """レイノルズ数"""
    return density * velocity * diameter / viscosity

def drag_force(coefficient, density, velocity, area):
    """抗力"""
    return 0.5 * coefficient * density * velocity * velocity * area

def centripetal_force(mass, velocity, radius):
    """向心力"""
    return mass * velocity * velocity / radius

def simple_harmonic_motion_period(mass, spring_constant):
    """単振動の周期"""
    return 2 * pi * sqrt(mass / spring_constant)

def pendulum_period(length, g=9.81):
    """単振り子の周期"""
    return 2 * pi * sqrt(length / g)

def wave_speed(frequency, wavelength):
    """波の速度"""
    return frequency * wavelength

def doppler_light(f0, v_relative, c_light=c):
    """光のドップラー効果"""
    return f0 * sqrt((1 + v_relative/c) / (1 - v_relative/c))

def time_dilation(t0, velocity):
    """時間の遅れ"""
    gamma = 1 / sqrt(1 - (velocity * velocity) / (c * c))
    return t0 * gamma

def length_contraction(l0, velocity):
    """長さの収縮"""
    gamma = 1 / sqrt(1 - (velocity * velocity) / (c * c))
    return l0 / gamma

def nuclear_binding_energy(mass_defect):
    """核結合エネルギー"""
    return mass_defect * c * c

def half_life_decay_constant(half_life):
    """半減期から崩壊定数"""
    return ln(2) / half_life

def radioactive_decay_remaining(n0, half_life, time):
    """放射性崩壊後の残存数"""
    lambda_val = half_life_decay_constant(half_life)
    return n0 * exp(-lambda_val * time)

def nuclear_reaction_energy(mass_reactants, mass_products):
    """核反応エネルギー"""
    mass_defect = mass_reactants - mass_products
    return mass_defect * c * c

def quantum_tunneling_probability(barrier_height, particle_energy, barrier_width, mass):
    """量子トンネル効果の確率（近似）"""
    k = sqrt(2 * mass * (barrier_height - particle_energy)) / hbar
    return exp(-2 * k * barrier_width)

def schrodinger_energy_infinite_well(n, length, mass):
    """無限井戸ポテンシャルのエネルギー"""
    return (h * h * n * n) / (8 * mass * length * length)

def heisenberg_uncertainty_momentum_position(delta_p):
    """ハイゼンベルクの不確定性原理（運動量・位置）"""
    return hbar / (2 * delta_p)

def compton_wavelength_shift(initial_wavelength, scattering_angle):
    """コンプトン効果による波長変化"""
    electron_mass = 9.10938356e-31  # kg
    return (h / (electron_mass * c)) * (1 - cos(scattering_angle))

def photoelectric_work_function(photon_energy, kinetic_energy):
    """光電効果の仕事関数"""
    return photon_energy - kinetic_energy

def zeeman_effect_energy(magnetic_field, orbital_angular_momentum):
    """ゼーマン効果によるエネルギー変化"""
    return mu_B * magnetic_field * orbital_angular_momentum

def nuclear_magnetic_resonance_frequency(gyromagnetic_ratio, magnetic_field):
    """核磁気共鳴周波数"""
    return gyromagnetic_ratio * magnetic_field / (2 * pi)

def superconductivity_critical_temperature(critical_field, critical_temperature_0):
    """超伝導の臨界温度"""
    # BCS理論の近似
    return critical_temperature_0 * sqrt(1 - critical_field / critical_field)

def fermi_energy(electron_density):
    """フェルミエネルギー"""
    electron_mass = 9.10938356e-31  # kg
    return (hbar * hbar / (2 * electron_mass)) * pow(3 * pi * pi * electron_density, 2/3)

def bose_einstein_condensation_temperature(particle_density, mass):
    """ボース・アインシュタイン凝縮の臨界温度"""
    return (hbar * hbar / (2 * mass)) * pow(particle_density / 2.612, 2/3)

def quantum_harmonic_oscillator_energy(n, frequency):
    """量子調和振動子のエネルギー"""
    return hbar * frequency * (n + 0.5)

def particle_in_box_energy(n, length, mass):
    """箱の中の粒子のエネルギー"""
    return (h * h * n * n) / (8 * mass * length * length)

def hydrogen_atom_energy(n):
    """水素原子のエネルギー準位"""
    return -13.6 / (n * n)  # eV

def hydrogen_atom_radius(n):
    """水素原子の軌道半径"""
    return bohr_radius * n * n

def molecular_vibration_frequency(force_constant, reduced_mass):
    """分子振動の周波数"""
    return (1 / (2 * pi)) * sqrt(force_constant / reduced_mass)

def molecular_rotation_energy(j, moment_of_inertia):
    """分子回転のエネルギー"""
    return (hbar * hbar * j * (j + 1)) / (2 * moment_of_inertia)

def crystal_lattice_energy(charges, distance, madelung_constant):
    """結晶格子エネルギー"""
    return madelung_constant * charges * charges / (4 * pi * eps0 * distance)

def semiconductor_band_gap_temperature(band_gap_0, temperature, alpha, beta):
    """半導体のバンドギャップ温度依存性"""
    return band_gap_0 - (alpha * temperature * temperature) / (temperature + beta)

def laser_gain_coefficient(stimulated_emission, absorption, population_inversion):
    """レーザー増幅係数"""
    return stimulated_emission - absorption * population_inversion

def optical_fiber_numerical_aperture(n_core, n_cladding):
    """光ファイバーの開口数"""
    return sqrt(n_core * n_core - n_cladding * n_cladding)

def diffraction_grating_angle(wavelength, grating_spacing, order):
    """回折格子の回折角"""
    return arcsin(order * wavelength / grating_spacing)

def thin_lens_focal_length(refractive_index, radius1, radius2, thickness):
    """薄いレンズの焦点距離"""
    return 1 / ((refractive_index - 1) * (1/radius1 - 1/radius2 + 
            (refractive_index - 1) * thickness / (refractive_index * radius1 * radius2)))

def telescope_magnification(focal_length_objective, focal_length_eyepiece):
    """望遠鏡の倍率"""
    return focal_length_objective / focal_length_eyepiece

def microscope_magnification(magnification_objective, magnification_eyepiece, tube_length, focal_length_objective):
    """顕微鏡の倍率"""
    return magnification_objective * magnification_eyepiece * tube_length / focal_length_objective

def interferometer_path_difference(wavelength, phase_difference):
    """干渉計の光路差"""
    return wavelength * phase_difference / (2 * pi)

def polarizer_malus_law(initial_intensity, angle):
    """マリュスの法則による偏光強度"""
    return initial_intensity * cos(angle) * cos(angle)

def magnetic_moment_spin(spin_quantum_number, g_factor):
    """スピン磁気モーメント"""
    return g_factor * mu_B * sqrt(spin_quantum_number * (spin_quantum_number + 1))

def nuclear_spin_energy(magnetic_field, nuclear_spin, g_factor):
    """核スピンエネルギー"""
    return g_factor * mu_N * magnetic_field * nuclear_spin

def quantum_tunneling_current(barrier_height, applied_voltage, barrier_width, mass):
    """量子トンネル電流"""
    k = sqrt(2 * mass * (barrier_height - applied_voltage)) / hbar
    transmission = exp(-2 * k * barrier_width)
    return transmission  # 簡略化された形

def josephson_junction_voltage(frequency):
    """ジョセフソン接合の電圧"""
    return (h / (2 * qe)) * frequency

def hall_effect_voltage(magnetic_field, current, thickness, charge_carrier_density):
    """ホール効果の電圧"""
    return magnetic_field * current / (charge_carrier_density * qe * thickness)

def quantum_hall_resistance(landau_level):
    """量子ホール抵抗"""
    return h / (qe * qe * landau_level)

def superconductivity_coherence_length(fermi_velocity, energy_gap):
    """超伝導のコヒーレンス長"""
    return hbar * fermi_velocity / (pi * energy_gap)

def bose_einstein_distribution(energy, temperature, chemical_potential):
    """ボース・アインシュタイン分布"""
    return 1 / (exp((energy - chemical_potential) / (k * temperature)) - 1)

def fermi_dirac_distribution(energy, temperature, fermi_energy):
    """フェルミ・ディラック分布"""
    return 1 / (exp((energy - fermi_energy) / (k * temperature)) + 1)

def maxwell_boltzmann_distribution(velocity, temperature, mass):
    """マクスウェル・ボルツマン分布"""
    return sqrt(mass / (2 * pi * k * temperature)) * exp(-mass * velocity * velocity / (2 * k * temperature))

def planck_radiation_law(wavelength, temperature):
    """プランクの放射則"""
    hc_lambda_kt = (h * c) / (wavelength * k * temperature)
    return (2 * pi * h * c * c) / pow(wavelength, 5) / (exp(hc_lambda_kt) - 1)

def rayleigh_jeans_law(wavelength, temperature):
    """レイリー・ジーンズの法則"""
    return (2 * pi * c * k * temperature) / pow(wavelength, 4)

def wien_approximation(wavelength, temperature):
    """ウィーンの近似"""
    hc_lambda_kt = (h * c) / (wavelength * k * temperature)
    return (2 * pi * h * c * c) / pow(wavelength, 5) * exp(-hc_lambda_kt)

# 工学・応用科学分野
def shannon_entropy(probabilities):
    """シャノンエントロピー"""
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy -= p * log2(p)
    return entropy

def mutual_information(p_xy, p_x, p_y):
    """相互情報量"""
    mi = 0
    for i in range(len(p_xy)):
        for j in range(len(p_xy[0])):
            if p_xy[i][j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i][j] * log2(p_xy[i][j] / (p_x[i] * p_y[j]))
    return mi

def channel_capacity(bandwidth, signal_power, noise_power):
    """シャノンの通信路容量"""
    return bandwidth * log2(1 + signal_power / noise_power)

def nyquist_rate(bandwidth):
    """ナイキストレート"""
    return 2 * bandwidth

def butterworth_filter_response(frequency, cutoff_frequency, order):
    """バターワースフィルタの応答"""
    return 1 / sqrt(1 + pow(frequency / cutoff_frequency, 2 * order))

def kalman_filter_prediction(x_prev, p_prev, f, q):
    """カルマンフィルタ予測ステップ"""
    x_pred = f * x_prev
    p_pred = f * p_prev * f + q
    return x_pred, p_pred

def kalman_filter_update(x_pred, p_pred, measurement, h, r):
    """カルマンフィルタ更新ステップ"""
    k = p_pred * h / (h * p_pred * h + r)  # カルマンゲイン
    x_update = x_pred + k * (measurement - h * x_pred)
    p_update = (1 - k * h) * p_pred
    return x_update, p_update

def pid_controller(error, prev_error, integral, kp, ki, kd, dt):
    """PID制御器"""
    derivative = (error - prev_error) / dt
    integral += error * dt
    output = kp * error + ki * integral + kd * derivative
    return output, integral

def fourier_transform_approximation(signal, frequencies):
    """フーリエ変換の近似"""
    result = []
    for freq in frequencies:
        real_part = 0
        imag_part = 0
        for i, sample in enumerate(signal):
            angle = 2 * pi * freq * i / len(signal)
            real_part += sample * cos(angle)
            imag_part += sample * sin(angle)
        result.append(Complex(real_part, imag_part))
    return result

def convolution_1d(signal, kernel):
    """1次元畳み込み"""
    result = []
    for i in range(len(signal) - len(kernel) + 1):
        sum_val = 0
        for j in range(len(kernel)):
            sum_val += signal[i + j] * kernel[j]
        result.append(sum_val)
    return result

def autocorrelation(signal, lag):
    """自己相関関数"""
    if lag >= len(signal):
        return 0
    sum_val = 0
    for i in range(len(signal) - lag):
        sum_val += signal[i] * signal[i + lag]
    return sum_val / (len(signal) - lag)

def cross_correlation(signal1, signal2, lag):
    """相互相関関数"""
    if lag >= len(signal1) or lag >= len(signal2):
        return 0
    sum_val = 0
    for i in range(min(len(signal1), len(signal2)) - lag):
        sum_val += signal1[i] * signal2[i + lag]
    return sum_val / (min(len(signal1), len(signal2)) - lag)

# 天文学・宇宙物理学
def hubble_law_distance(redshift, h0=70):
    """ハッブルの法則による距離"""
    return redshift * c / h0

def cosmological_redshift(z):
    """宇宙論的赤方偏移"""
    return sqrt((1 + z) * (1 + z) - 1)

def schwarzschild_radius(mass):
    """シュバルツシルト半径"""
    return 2 * G * mass / (c * c)

def gravitational_lensing_angle(mass, impact_parameter):
    """重力レンズ効果の偏角"""
    return 4 * G * mass / (c * c * impact_parameter)

def cosmic_microwave_background_temperature(redshift):
    """宇宙マイクロ波背景放射の温度"""
    return 2.725 * (1 + redshift)  # K

def dark_energy_density(omega_lambda, h0):
    """ダークエネルギー密度"""
    return omega_lambda * 3 * h0 * h0 / (8 * pi * G)

def stellar_luminosity(mass):
    """恒星の光度（質量光度関係）"""
    return pow(mass, 3.5)  # 太陽光度単位

def stellar_lifetime(mass, luminosity):
    """恒星の寿命"""
    return mass / luminosity  # 太陽年単位

def orbital_period_kepler(mass1, mass2, semi_major_axis):
    """ケプラーの第三法則による軌道周期"""
    return 2 * pi * sqrt(pow(semi_major_axis, 3) / (G * (mass1 + mass2)))

def doppler_shift_cosmological(redshift):
    """宇宙論的ドップラー効果"""
    return (1 + redshift) * (1 + redshift) - 1

def chandrasekhar_limit():
    """チャンドラセカール限界"""
    return 1.4  # 太陽質量

def eddington_luminosity(mass):
    """エディントン光度"""
    thomson_cross_section = 6.65e-29  # m²
    return 4 * pi * G * mass * c / thomson_cross_section

# 地球科学・気象学
def adiabatic_lapse_rate(gamma=1.4):
    """断熱減率"""
    return 9.8 / (1005 * gamma)  # K/km

def coriolis_force(latitude, velocity):
    """コリオリ力"""
    omega = 7.292e-5  # 地球の自転角速度
    return 2 * omega * velocity * sin(latitude)

def geostrophic_wind(pressure_gradient, latitude):
    """地衡風"""
    rho = 1.225  # 空気密度
    omega = 7.292e-5
    return pressure_gradient / (2 * rho * omega * sin(latitude))

def atmospheric_scale_height(temperature, molar_mass=0.029):
    """大気スケールハイト"""
    return R * temperature / (molar_mass * 9.81)

def seismic_wave_velocity(bulk_modulus, shear_modulus, density):
    """地震波速度"""
    # P波速度
    vp = sqrt((bulk_modulus + 4/3 * shear_modulus) / density)
    # S波速度
    vs = sqrt(shear_modulus / density)
    return vp, vs

def richter_magnitude(amplitude, distance):
    """リヒタースケール"""
    return log10(amplitude) + 3 * log10(distance) - 2.92

def moment_magnitude(moment):
    """モーメントマグニチュード"""
    return (2/3) * log10(moment) - 6.0

def plate_tectonics_velocity(age, spreading_rate):
    """プレートテクトニクス速度"""
    return spreading_rate / (2 * age)

# 生物物理学・神経科学
def hodgkin_huxley_na_conductance(v, t):
    """ホジキン・ハクスレーモデルNa+コンダクタンス"""
    alpha_m = 0.1 * (25 - v) / (exp((25 - v) / 10) - 1)
    beta_m = 4 * exp(-v / 18)
    m_inf = alpha_m / (alpha_m + beta_m)
    tau_m = 1 / (alpha_m + beta_m)
    return m_inf * (1 - exp(-t / tau_m))

def action_potential_frequency(input_current, threshold_current):
    """活動電位の頻度"""
    if input_current <= threshold_current:
        return 0
    return (input_current - threshold_current) / (threshold_current * 0.1)

def neural_synaptic_strength(initial_strength, learning_rate, activity):
    """シナプス強度の変化"""
    return initial_strength + learning_rate * activity * (1 - activity)

def population_growth_logistic(initial_population, carrying_capacity, growth_rate, time):
    """ロジスティック成長"""
    return carrying_capacity / (1 + (carrying_capacity - initial_population) / initial_population * exp(-growth_rate * time))

def enzyme_kinetics_michaelis_menten(substrate_concentration, vmax, km):
    """ミカエリス・メンテン式"""
    return vmax * substrate_concentration / (km + substrate_concentration)

def diffusion_coefficient(temperature, viscosity, particle_radius):
    """拡散係数（ストークス・アインシュタイン式）"""
    return k * temperature / (6 * pi * viscosity * particle_radius)

def membrane_potential_nernst(ion_concentration_outside, ion_concentration_inside, valence):
    """ネルンスト電位"""
    return (R * 310) / (valence * Faraday) * ln(ion_concentration_outside / ion_concentration_inside)

# 材料科学・ナノテクノロジー
def debye_temperature(debye_frequency):
    """デバイ温度"""
    return hbar * debye_frequency / k

def specific_heat_debye(temperature, debye_temperature):
    """デバイ比熱"""
    if temperature < debye_temperature:
        return 12 * pi * pi * pi * pi / 5 * pow(temperature / debye_temperature, 3)
    else:
        return 3 * R

def thermal_expansion_coefficient(gruneisen_parameter, bulk_modulus, specific_heat):
    """熱膨張係数"""
    return gruneisen_parameter * specific_heat / bulk_modulus

def young_modulus_from_atomic_parameters(bond_energy, bond_length, atomic_volume):
    """原子パラメータからのヤング率"""
    return bond_energy / (bond_length * atomic_volume)

def quantum_dot_energy_level(confinement_length, effective_mass):
    """量子ドットのエネルギー準位"""
    return (hbar * hbar * pi * pi) / (2 * effective_mass * confinement_length * confinement_length)

def carbon_nanotube_band_gap(diameter, chiral_angle):
    """カーボンナノチューブのバンドギャップ"""
    # 簡略化されたモデル
    return 2.7 / diameter  # eV

def graphene_fermi_velocity():
    """グラフェンのフェルミ速度"""
    return 1e6  # m/s

def plasmon_frequency(electron_density):
    """プラズモン周波数"""
    return sqrt(4 * pi * electron_density * qe * qe / (9.11e-31))  # 電子質量

def magnetic_anisotropy_energy(anisotropy_constant, volume):
    """磁気異方性エネルギー"""
    return anisotropy_constant * volume

# 金融工学・経済物理学
def black_scholes_call_price(s0, k, t, r, sigma):
    """ブラック・ショールズ・コールオプション価格"""
    d1 = (ln(s0/k) + (r + 0.5*sigma*sigma)*t) / (sigma*sqrt(t))
    d2 = d1 - sigma*sqrt(t)
    return s0*normal_cdf(d1) - k*exp(-r*t)*normal_cdf(d2)

def normal_cdf(x):
    """標準正規分布の累積分布関数（近似）"""
    return 0.5 * (1 + erf(x / sqrt(2)))

def erf(x):
    """誤差関数（近似）"""
    # 簡略化された近似
    return 2 / sqrt(pi) * x * exp(-x*x)

def value_at_risk(portfolio_value, volatility, confidence_level, time_horizon):
    """バリュー・アット・リスク"""
    z_score = 1.96  # 95%信頼区間
    return portfolio_value * z_score * volatility * sqrt(time_horizon)

def sharpe_ratio(return_rate, risk_free_rate, volatility):
    """シャープレシオ"""
    return (return_rate - risk_free_rate) / volatility

def power_law_distribution(x, alpha, x_min):
    """べき乗分布"""
    return (alpha - 1) * pow(x_min, alpha - 1) * pow(x, -alpha)

def hurst_exponent(time_series):
    """ハースト指数"""
    n = len(time_series)
    rs_values = []
    for m in range(10, n//2):
        k = n // m
        rs = 0
        for i in range(k):
            segment = time_series[i*m:(i+1)*m]
            mean_val = mean(segment)
            std_val = stddev(segment)
            if std_val > 0:
                rs += (max(segment) - min(segment)) / std_val
        rs_values.append(rs / k)
    return mean([ln(rs) for rs in rs_values])

# 機械学習・人工知能
def sigmoid_function(x):
    """シグモイド関数"""
    return 1 / (1 + exp(-x))

def relu_function(x):
    """ReLU関数"""
    return max(0, x)

def softmax_function(logits):
    """ソフトマックス関数"""
    exp_logits = [exp(x) for x in logits]
    sum_exp = sum(exp_logits)
    return [x / sum_exp for x in exp_logits]

def cross_entropy_loss(predictions, targets):
    """交差エントロピー損失"""
    loss = 0
    for p, t in zip(predictions, targets):
        if p > 0:
            loss -= t * ln(p)
    return loss

def gradient_descent_update(parameter, gradient, learning_rate):
    """勾配降下法の更新"""
    return parameter - learning_rate * gradient

def momentum_update(parameter, gradient, velocity, momentum, learning_rate):
    """モーメンタム更新"""
    velocity = momentum * velocity + learning_rate * gradient
    return parameter - velocity, velocity

# 量子コンピューティング
def quantum_gate_hadamard():
    """アダマールゲート"""
    return [[1, 1], [1, -1]] / sqrt(2)

def quantum_gate_cnot():
    """CNOTゲート"""
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]

def quantum_entanglement_measurement(state_vector):
    """量子もつれの測定（簡略化）"""
    # ベル状態の例
    if len(state_vector) == 4:
        return abs(state_vector[0] * state_vector[3] - state_vector[1] * state_vector[2])
    return 0

def quantum_fourier_transform_phase(qubit_index, total_qubits):
    """量子フーリエ変換の位相"""
    return 2 * pi * qubit_index / pow(2, total_qubits)

def grover_algorithm_iterations(n_items):
    """グローバーアルゴリズムの反復回数"""
    return int(pi / 4 * sqrt(n_items))

def quantum_teleportation_fidelity(entanglement_measure):
    """量子テレポーテーションの忠実度"""
    return 0.5 + 0.5 * entanglement_measure

# 宇宙生物学・地球外生命
def drake_equation(star_formation_rate, fraction_with_planets, fraction_habitable, fraction_life, fraction_intelligent, fraction_communicative, civilization_lifetime):
    """ドレーク方程式"""
    return star_formation_rate * fraction_with_planets * fraction_habitable * fraction_life * fraction_intelligent * fraction_communicative * civilization_lifetime

def habitable_zone_distance(star_luminosity, star_temperature):
    """ハビタブルゾーンの距離"""
    return sqrt(star_luminosity / 1.0)  # 太陽光度単位

def atmospheric_escape_velocity(planet_mass, planet_radius, temperature, molecular_mass):
    """大気脱出速度"""
    escape_velocity = sqrt(2 * G * planet_mass / planet_radius)
    thermal_velocity = sqrt(3 * k * temperature / molecular_mass)
    return escape_velocity / thermal_velocity

def biosignature_detection_signal_to_noise(planet_radius, star_radius, orbital_distance, atmospheric_scale_height):
    """バイオシグネチャ検出の信号雑音比"""
    transit_depth = pow(planet_radius / star_radius, 2)
    atmospheric_signal = transit_depth * (atmospheric_scale_height / planet_radius)
    return atmospheric_signal

# ナノメディシン・薬物送達
def nanoparticle_diffusion_coefficient(particle_radius, temperature, viscosity):
    """ナノ粒子の拡散係数"""
    return k * temperature / (6 * pi * viscosity * particle_radius)

def drug_release_rate_controlled(diffusion_coefficient, concentration_gradient, membrane_thickness):
    """制御薬物放出速度"""
    return diffusion_coefficient * concentration_gradient / membrane_thickness

def targeted_drug_delivery_efficiency(ligand_concentration, receptor_concentration, binding_affinity):
    """標的薬物送達効率"""
    return ligand_concentration * receptor_concentration / (binding_affinity + ligand_concentration)

def cancer_cell_growth_rate(initial_cells, doubling_time, time):
    """がん細胞増殖速度"""
    return initial_cells * pow(2, time / doubling_time)

def chemotherapy_dose_response(cell_survival, drug_concentration, ic50):
    """化学療法の用量反応"""
    return 1 / (1 + pow(drug_concentration / ic50, 2))

# 気候変動・環境科学
def radiative_forcing_co2(co2_concentration, co2_reference=280):
    """CO2による放射強制力"""
    return 5.35 * ln(co2_concentration / co2_reference)

def global_warming_potential(gas_lifetime, radiative_efficiency, co2_radiative_efficiency=1.37e-5):
    """地球温暖化係数"""
    return (radiative_efficiency / co2_radiative_efficiency) * gas_lifetime

def ocean_acidification_ph_change(atmospheric_co2, preindustrial_co2=280):
    """海洋酸性化によるpH変化"""
    return -0.0031 * (atmospheric_co2 - preindustrial_co2)

def sea_level_rise_thermal_expansion(temperature_increase, ocean_depth=4000):
    """熱膨張による海面上昇"""
    thermal_expansion_coefficient = 2.1e-4  # 1/K
    return thermal_expansion_coefficient * temperature_increase * ocean_depth

def carbon_cycle_atmospheric_lifetime(anthropogenic_emissions, natural_sinks):
    """炭素循環の大気中滞留時間"""
    return anthropogenic_emissions / natural_sinks

# エネルギー・持続可能性
def solar_panel_efficiency_theoretical(band_gap_energy):
    """太陽電池の理論効率（ショックレー・クワイサー限界）"""
    return 0.33 * band_gap_energy / 1.1  # 1.1 eVを基準とした簡略化

def wind_turbine_power_output(air_density, swept_area, wind_speed, power_coefficient=0.4):
    """風力タービンの出力"""
    return 0.5 * air_density * swept_area * pow(wind_speed, 3) * power_coefficient

def nuclear_fusion_energy_output(deuterium_density, tritium_density, temperature, confinement_time):
    """核融合エネルギー出力（ローソン条件）"""
    return deuterium_density * tritium_density * temperature * confinement_time

def battery_energy_density(cell_voltage, capacity):
    """バッテリーのエネルギー密度"""
    return cell_voltage * capacity

def hydrogen_fuel_cell_efficiency(standard_enthalpy, gibbs_free_energy):
    """水素燃料電池の効率"""
    return gibbs_free_energy / standard_enthalpy

# 交通工学・都市計画
def traffic_flow_greenshields(density, jam_density, free_flow_speed):
    """グリーンシールズモデルによる交通流"""
    return free_flow_speed * density * (1 - density / jam_density)

def urban_heat_island_temperature_difference(urban_area, rural_area, heat_flux):
    """都市ヒートアイランド効果"""
    return heat_flux * urban_area / (rural_area * 1000)  # 簡略化

def public_transport_accessibility(population_density, station_density, walking_speed=1.4):
    """公共交通アクセシビリティ"""
    return population_density * station_density / (walking_speed * 1000)

def building_energy_efficiency(thermal_conductivity, wall_thickness, temperature_difference):
    """建物のエネルギー効率"""
    return thermal_conductivity * temperature_difference / wall_thickness

# 音響学・音楽理論
def musical_scale_frequency(base_frequency, semitone_steps):
    """音楽スケールの周波数"""
    return base_frequency * pow(2, semitone_steps / 12)

def doppler_effect_audio(source_frequency, source_velocity, observer_velocity, sound_speed=343):
    """音のドップラー効果"""
    return source_frequency * (sound_speed + observer_velocity) / (sound_speed - source_velocity)

def room_acoustics_reverberation_time(room_volume, total_absorption):
    """室内音響の残響時間（サビン公式）"""
    return 0.161 * room_volume / total_absorption

def musical_harmonic_series(fundamental_frequency, harmonic_number):
    """音楽の倍音系列"""
    return fundamental_frequency * harmonic_number

def equal_temperament_interval(interval_semitones):
    """平均律の音程"""
    return pow(2, interval_semitones / 12)

# スポーツ科学・運動生理学
def vo2_max_prediction(age, weight, resting_heart_rate, max_heart_rate):
    """最大酸素摂取量の予測"""
    return 15.3 * max_heart_rate / resting_heart_rate - 0.15 * age - 0.1 * weight + 50

def lactate_threshold_heart_rate(vo2_max, training_status):
    """乳酸閾値心拍数"""
    return 0.85 * (220 - 20) + training_status * 5  # 簡略化

def muscle_force_length_relationship(optimal_length, current_length):
    """筋力-長さ関係"""
    return exp(-pow((current_length - optimal_length) / (0.5 * optimal_length), 2))

def running_economy(oxygen_cost, running_speed):
    """ランニングエコノミー"""
    return oxygen_cost / running_speed

def power_output_cycling(cadence, torque):
    """サイクリングのパワー出力"""
    return 2 * pi * cadence * torque / 60

# 食品科学・栄養学
def maillard_reaction_rate(temperature, moisture_content, ph):
    """メイラード反応速度"""
    return exp(-10000 / (R * temperature)) * moisture_content * (7 - ph)

def food_shelf_life_prediction(initial_quality, temperature, water_activity, oxygen_level):
    """食品の賞味期限予測"""
    return initial_quality * exp(-0.1 * temperature * water_activity * oxygen_level)

def nutrient_bioavailability(initial_concentration, absorption_efficiency, metabolism_rate):
    """栄養素の生体利用率"""
    return initial_concentration * absorption_efficiency * (1 - metabolism_rate)

def food_energy_density(protein_content, fat_content, carbohydrate_content):
    """食品のエネルギー密度"""
    return 4 * protein_content + 9 * fat_content + 4 * carbohydrate_content

def fermentation_kinetics(substrate_concentration, max_growth_rate, saturation_constant):
    """発酵反応速度"""
    return max_growth_rate * substrate_concentration / (saturation_constant + substrate_concentration)

# 考古学・年代測定
def radiocarbon_dating_age(carbon14_ratio, half_life=5730):
    """放射性炭素年代測定"""
    return -half_life * ln(carbon14_ratio) / ln(2)

def thermoluminescence_dating(dose_rate, accumulated_dose):
    """熱ルミネッセンス年代測定"""
    return accumulated_dose / dose_rate

def dendrochronology_correlation(tree_ring_series1, tree_ring_series2):
    """年輪年代学の相関"""
    return cross_correlation(tree_ring_series1, tree_ring_series2, 0)

def potassium_argon_dating(potassium40_ratio, argon40_ratio, decay_constant=5.81e-11):
    """カリウム・アルゴン年代測定"""
    return ln(1 + argon40_ratio / potassium40_ratio) / decay_constant

def uranium_lead_dating(uranium238_ratio, lead206_ratio, half_life=4.47e9):
    """ウラン・鉛年代測定"""
    return -half_life * ln(1 - lead206_ratio / uranium238_ratio) / ln(2)
