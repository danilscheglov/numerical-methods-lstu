import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QGridLayout,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import time


class SolverApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Решение нелинейных уравнений")
        self.setGeometry(100, 100, 1000, 700)

        self.equations = {
            "4*x*ln(x)^2 - 4*sqrt(1+x) + 5 = 0": "4*x*np.log(x)**2 - 4*np.sqrt(1 + x) + 5",
            "x^3 - 6*x^2 + 11*x - 6 = 0": "x**3 - 6*x**2 + 11*x - 6",
            "cos(x) - x = 0": "np.cos(x) - x",
            "x^2 - 4*sin(x) = 0": "x**2 - 4*np.sin(x)",
        }

        self.recommended_intervals = {
            "4*x*ln(x)^2 - 4*sqrt(1+x) + 5 = 0": (0.5, 1.5),
            "x^3 - 6*x^2 + 11*x - 6 = 0": (1, 4),
            "cos(x) - x = 0": (0, 1),
            "x^2 - 4*sin(x) = 0": (1.5, 2.5),
        }

        self.method_names = {
            0: "Метод половинного деления",
            1: "Метод хорд",
            2: "Метод Ньютона",
            3: "Метод секущих",
            4: "Гибридный метод",
        }

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.create_ui()

    def create_ui(self):
        main_layout = QVBoxLayout(self.central_widget)

        input_widget = QWidget()
        input_layout = QGridLayout(input_widget)

        input_layout.addWidget(QLabel("Выберите уравнение:"), 0, 0)
        self.equation_combo = QComboBox()
        self.equation_combo.addItems(list(self.equations.keys()))
        self.equation_combo.currentIndexChanged.connect(self.update_equation)
        input_layout.addWidget(self.equation_combo, 0, 1)

        input_layout.addWidget(QLabel("Функция f(x):"), 1, 0)
        self.func_input = QLineEdit(self.equations[list(self.equations.keys())[0]])
        input_layout.addWidget(self.func_input, 1, 1)

        input_layout.addWidget(QLabel("Левая граница (a):"), 2, 0)
        self.a_input = QLineEdit(
            str(self.recommended_intervals[list(self.equations.keys())[0]][0])
        )
        input_layout.addWidget(self.a_input, 2, 1)

        input_layout.addWidget(QLabel("Правая граница (b):"), 3, 0)
        self.b_input = QLineEdit(
            str(self.recommended_intervals[list(self.equations.keys())[0]][1])
        )
        input_layout.addWidget(self.b_input, 3, 1)

        input_layout.addWidget(QLabel("Точность (ε):"), 4, 0)
        self.epsilon_input = QLineEdit("1e-12")
        input_layout.addWidget(self.epsilon_input, 4, 1)

        input_layout.addWidget(QLabel("Максимальное число итераций:"), 5, 0)
        self.max_iter_input = QLineEdit("100")
        input_layout.addWidget(self.max_iter_input, 5, 1)

        input_layout.addWidget(QLabel("Метод решения:"), 6, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(
            [
                "Метод половинного деления",
                "Метод хорд",
                "Метод Ньютона",
                "Метод секущих",
                "Гибридный метод",
            ]
        )
        input_layout.addWidget(self.method_combo, 6, 1)

        button_layout = QHBoxLayout()
        self.solve_button = QPushButton("Решить")
        self.solve_button.clicked.connect(self.solve_equation)
        self.plot_button = QPushButton("Построить график")
        self.plot_button.clicked.connect(self.plot_function)
        button_layout.addWidget(self.solve_button)
        button_layout.addWidget(self.plot_button)
        input_layout.addLayout(button_layout, 7, 0, 1, 2)

        main_layout.addWidget(input_widget)

        self.tabs = QTabWidget()

        # --- Результаты ---
        self.result_tab = QWidget()
        result_layout = QVBoxLayout(self.result_tab)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(
            ["Итерация", "x", "|x_k - x_{k-1}|", "|f(x)|"]
        )
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        result_layout.addWidget(self.result_table)

        info_layout = QGridLayout()
        info_layout.addWidget(QLabel("Найденный корень:"), 0, 0)
        self.root_label = QLabel("-")
        info_layout.addWidget(self.root_label, 0, 1)

        info_layout.addWidget(QLabel("Значение функции в корне:"), 1, 0)
        self.func_value_label = QLabel("-")
        info_layout.addWidget(self.func_value_label, 1, 1)

        info_layout.addWidget(QLabel("Число итераций:"), 2, 0)
        self.iterations_label = QLabel("-")
        info_layout.addWidget(self.iterations_label, 2, 1)

        info_layout.addWidget(QLabel("Погрешность:"), 3, 0)
        self.error_label = QLabel("-")
        info_layout.addWidget(self.error_label, 3, 1)

        info_layout.addWidget(QLabel("Метод:"), 4, 0)
        self.method_label = QLabel("-")
        info_layout.addWidget(self.method_label, 4, 1)

        info_layout.addWidget(QLabel("Время выполнения:"), 5, 0)
        self.time_label = QLabel("-")
        info_layout.addWidget(self.time_label, 5, 1)

        result_layout.addLayout(info_layout)
        self.tabs.addTab(self.result_tab, "Результаты")

        # --- График ---
        self.plot_tab = QWidget()
        plot_layout = QVBoxLayout(self.plot_tab)
        self.figure = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        self.tabs.addTab(self.plot_tab, "График")

        # --- Сравнение методов ---
        self.comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(self.comparison_tab)

        self.comparison_button = QPushButton("Выполнить сравнительный анализ")
        self.comparison_button.clicked.connect(self.compare_methods)
        comparison_layout.addWidget(self.comparison_button)

        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(4)  # Уменьшено до 4 столбцов
        self.comparison_table.setHorizontalHeaderLabels(
            ["Метод", "Корень", "Итерации", "Время (мс)"]
        )  # Убрали f(x)
        self.comparison_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.comparison_table.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.comparison_table.setMinimumHeight(400)
        comparison_layout.addWidget(self.comparison_table)

        self.comparison_figure = plt.figure(figsize=(10, 6))
        self.comparison_canvas = FigureCanvas(self.comparison_figure)
        comparison_layout.addWidget(self.comparison_canvas)

        self.tabs.addTab(self.comparison_tab, "Сравнение")

        main_layout.addWidget(self.tabs)

    def update_equation(self):
        equation = self.equation_combo.currentText()
        self.func_input.setText(self.equations[equation])
        a, b = self.recommended_intervals[equation]
        self.a_input.setText(str(a))
        self.b_input.setText(str(b))

    def evaluate_function(self, x):
        func_str = self.func_input.text()
        try:
            result = eval(func_str)
            if np.isnan(result) or np.isinf(result):
                raise ValueError("Функция возвращает NaN или inf")
            return result
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка вычисления функции: {str(e)}")
            return None

    def derivative(self, x, h=1e-10):
        fx_right = self.evaluate_function(x + h)
        fx_left = self.evaluate_function(x)
        if fx_right is None or fx_left is None:
            return None
        return (fx_right - fx_left) / h

    def fourier_analysis(self, a, b, num_points=2000):
        try:
            x = np.linspace(a, b, num_points)
            y = [self.evaluate_function(xi) for xi in x]
            if any(val is None for val in y) or len(y) == 0:
                raise ValueError("Ошибка при построении спектра")
            spectrum = np.fft.fft(y)
            freq = np.fft.fftfreq(len(x), (b - a) / num_points)
            amplitudes = np.abs(spectrum) / num_points
            positive_freq = freq[: num_points // 2]
            positive_amplitudes = amplitudes[: num_points // 2]
            dominant_freq_index = np.argmax(positive_amplitudes[1:]) + 1
            dominant_freq = abs(positive_freq[dominant_freq_index])
            return {"dominant_freq": dominant_freq}
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка анализа Фурье: {str(e)}")
            return None

    def bisection_method(self, a, b, epsilon, max_iter):
        iterations = []
        fa = self.evaluate_function(a)
        fb = self.evaluate_function(b)
        if fa is None or fb is None or fa * fb >= 0:
            return None, iterations
        for i in range(max_iter):
            c = (a + b) / 2
            fc = self.evaluate_function(c)
            error = abs(fc)
            iterations.append(
                {"iter": i + 1, "x": c, "fx": fc, "error": error, "abs_fx": abs(fc)}
            )
            if abs(fc) < epsilon or error < epsilon:
                return c, iterations
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        return (a + b) / 2, iterations

    def chord_method(self, a, b, epsilon, max_iter):
        iterations = []
        fa = self.evaluate_function(a)
        fb = self.evaluate_function(b)
        if fa is None or fb is None or fa * fb >= 0:
            return None, iterations
        for i in range(max_iter):
            if abs(fb - fa) < 1e-10:
                return None, iterations
            c = (a * fb - b * fa) / (fb - fa)
            fc = self.evaluate_function(c)
            error = abs(fc)
            iterations.append(
                {"iter": i + 1, "x": c, "fx": fc, "error": error, "abs_fx": abs(fc)}
            )
            if abs(fc) < epsilon or error < epsilon:
                return c, iterations
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        return c, iterations

    def newton_method(self, a, b, epsilon, max_iter):
        iterations = []
        x = (a + b) / 2
        for i in range(max_iter):
            fx = self.evaluate_function(x)
            dfx = self.derivative(x)
            if dfx is None or abs(dfx) < 1e-10:
                x = (a + b) / 2
                fx = self.evaluate_function(x)
            x_new = x - fx / dfx
            if x_new < a or x_new > b:
                return self.bisection_method(a, b, epsilon, max_iter)
            error = abs(fx)
            iterations.append(
                {"iter": i + 1, "x": x_new, "fx": fx, "error": error, "abs_fx": abs(fx)}
            )
            if abs(fx) < epsilon or error < epsilon:
                return x_new, iterations
            x = x_new
        return x, iterations

    def secant_method(self, a, b, epsilon, max_iter):
        iterations = []
        x0 = (a + b) / 2 - (b - a) / 4
        x1 = (a + b) / 2 + (b - a) / 4
        fx0 = self.evaluate_function(x0)
        fx1 = self.evaluate_function(x1)
        if fx0 is None or fx1 is None:
            return None, iterations

        fourier_result = self.fourier_analysis(a, b)
        use_secant = True
        if fourier_result:
            dominant_freq = fourier_result["dominant_freq"]
            threshold = 1e6 if (b - a) == 0 else 10 / (b - a)
            if dominant_freq > threshold * 2:
                reply = QMessageBox.question(
                    self,
                    "Подтверждение",
                    "Функция имеет высокочастотные компоненты. Метод секущих может быть неустойчив. "
                    "Хотите использовать метод половинного деления вместо него?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    use_secant = False

        if not use_secant:
            return self.bisection_method(a, b, epsilon, max_iter)

        for i in range(max_iter):
            if abs(fx1 - fx0) < 1e-10:
                return self.bisection_method(a, b, epsilon, max_iter)
            x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            if x_new < a or x_new > b:
                return self.bisection_method(a, b, epsilon, max_iter)
            fx_new = self.evaluate_function(x_new)
            error = abs(fx_new)
            iterations.append(
                {
                    "iter": i + 1,
                    "x": x_new,
                    "fx": fx_new,
                    "error": error,
                    "abs_fx": abs(fx_new),
                }
            )
            if abs(fx_new) < epsilon or error < epsilon:
                return x_new, iterations
            x0, x1 = x1, x_new
            fx0, fx1 = fx1, fx_new
        return x1, iterations

    def hybrid_method(self, a, b, epsilon, max_iter):
        iterations = []
        x_prev = None
        fa = self.evaluate_function(a)
        fb = self.evaluate_function(b)
        if fa is None or fb is None or fa * fb >= 0:
            QMessageBox.warning(
                self,
                "Предупреждение",
                "Функция должна иметь разные знаки на концах интервала для гибридного метода",
            )
            return None, iterations
        x = (a + b) / 2
        for i in range(max_iter):
            fx = self.evaluate_function(x)
            if fx is None:
                return None, iterations
            dfx = self.derivative(x)
            if abs(dfx) > 1e-10:
                x_newton = x - fx / dfx
                if a <= x_newton <= b:
                    x_new = x_newton
                else:
                    x_new = (a + b) / 2
            else:
                x_new = (a + b) / 2
            fx_new = self.evaluate_function(x_new)
            if fx_new is None:
                return None, iterations
            error = abs(x_new - x) if x_prev is not None else abs(b - a) / 2
            iterations.append(
                {
                    "iter": i + 1,
                    "x": x_new,
                    "fx": fx_new,
                    "error": error,
                    "abs_fx": abs(fx_new),
                }
            )
            if abs(fx_new) < epsilon or error < epsilon:
                return x_new, iterations
            if fa * fx_new < 0:
                b = x_new
                fb = fx_new
            else:
                a = x_new
                fa = fx_new
            x_prev = x
            x = x_new
        QMessageBox.warning(
            self, "Предупреждение", f"Гибридный метод не сошелся за {max_iter} итераций"
        )
        return x, iterations

    def solve_equation(self):
        try:
            a = float(self.a_input.text())
            b = float(self.b_input.text())
            epsilon = float(self.epsilon_input.text())
            max_iter = int(self.max_iter_input.text())
            method_index = self.method_combo.currentIndex()
            start_time = time.time()

            if method_index == 0:
                root, iterations = self.bisection_method(a, b, epsilon, max_iter)
            elif method_index == 1:
                root, iterations = self.chord_method(a, b, epsilon, max_iter)
            elif method_index == 2:
                root, iterations = self.newton_method(a, b, epsilon, max_iter)
            elif method_index == 3:
                root, iterations = self.secant_method(a, b, epsilon, max_iter)
            elif method_index == 4:
                root, iterations = self.hybrid_method(a, b, epsilon, max_iter)

            end_time = time.time()
            calc_time = (end_time - start_time) * 1000

            if root is not None:
                self.display_results(root, iterations, calc_time, method_index)
                self.tabs.setCurrentIndex(0)
                self.plot_function()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}")

    def display_results(self, root, iterations, calc_time, method_index):
        self.result_table.setRowCount(len(iterations))
        for i, iteration in enumerate(iterations):
            self.result_table.setItem(i, 0, QTableWidgetItem(str(iteration["iter"])))
            self.result_table.setItem(i, 1, QTableWidgetItem(f"{iteration['x']:.12e}"))
            self.result_table.setItem(
                i, 2, QTableWidgetItem(f"{iteration['error']:.12e}")
            )
            self.result_table.setItem(
                i, 3, QTableWidgetItem(f"{iteration['abs_fx']:.12e}")
            )

        self.root_label.setText(f"{root:.12e}")
        f_root = self.evaluate_function(root)
        self.func_value_label.setText(f"{f_root:.12e}")
        self.iterations_label.setText(str(len(iterations)))
        if len(iterations) > 0:
            self.error_label.setText(f"{iterations[-1]['error']:.12e}")
        self.method_label.setText(self.method_names[method_index])
        self.time_label.setText(f"{calc_time:.2f} мс")

    def plot_function(self):
        try:
            a = float(self.a_input.text())
            b = float(self.b_input.text())
            interval_width = b - a
            plot_a = a - 0.1 * interval_width
            plot_b = b + 0.1 * interval_width
            x = np.linspace(plot_a, plot_b, 1000)
            y = [self.evaluate_function(xi) for xi in x]

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(x, y, "b-", linewidth=2)
            ax.axhline(y=0, color="r", linestyle="--")
            ax.axvline(x=a, color="g", linestyle="--", alpha=0.5)
            ax.axvline(x=b, color="g", linestyle="--", alpha=0.5)

            if self.root_label.text() != "-":
                root = float(self.root_label.text())
                ax.plot(root, 0, "ro", markersize=8)
                ax.annotate(
                    f"Корень: {root:.6f}",
                    xy=(root, 0),
                    xytext=(root, 0.1),
                    arrowprops=dict(facecolor="black", shrink=0.05),
                )

            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title("График функции")
            ax.grid(True)
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка построения графика: {str(e)}")

    def compare_methods(self):
        try:
            a = float(self.a_input.text())
            b = float(self.b_input.text())
            epsilon = float(self.epsilon_input.text())
            max_iter = int(self.max_iter_input.text())

            methods = [
                (
                    "Метод половинного деления",
                    lambda: self.bisection_method(a, b, epsilon, max_iter),
                ),
                ("Метод хорд", lambda: self.chord_method(a, b, epsilon, max_iter)),
                ("Метод Ньютона", lambda: self.newton_method(a, b, epsilon, max_iter)),
                ("Метод секущих", lambda: self.secant_method(a, b, epsilon, max_iter)),
                (
                    "Гибридный метод",
                    lambda: self.hybrid_method(a, b, epsilon, max_iter),
                ),
            ]

            self.comparison_table.setRowCount(len(methods))
            for i, (method_name, method_func) in enumerate(methods):
                try:
                    start_time = time.time()
                    root, iterations = method_func()
                    end_time = time.time()
                    calc_time = (end_time - start_time) * 1000

                    if root is not None:
                        self.comparison_table.setItem(
                            i, 0, QTableWidgetItem(method_name)
                        )
                        self.comparison_table.setItem(
                            i, 1, QTableWidgetItem(f"{root:.12e}")
                        )
                        self.comparison_table.setItem(
                            i, 2, QTableWidgetItem(str(len(iterations)))
                        )
                        self.comparison_table.setItem(
                            i, 3, QTableWidgetItem(f"{calc_time:.2f}")
                        )
                    else:
                        self.comparison_table.setItem(
                            i, 0, QTableWidgetItem(method_name)
                        )
                        self.comparison_table.setItem(
                            i, 1, QTableWidgetItem("Не сошёлся")
                        )
                        self.comparison_table.setItem(i, 2, QTableWidgetItem("-"))
                        self.comparison_table.setItem(
                            i, 3, QTableWidgetItem(f"{calc_time:.2f}")
                        )
                except Exception as e:
                    self.comparison_table.setItem(i, 0, QTableWidgetItem(method_name))
                    self.comparison_table.setItem(
                        i, 1, QTableWidgetItem(f"Ошибка: {str(e)}")
                    )
                    self.comparison_table.setItem(i, 2, QTableWidgetItem("-"))
                    self.comparison_table.setItem(i, 3, QTableWidgetItem("-"))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка сравнения: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SolverApp()
    window.show()
    sys.exit(app.exec_())
