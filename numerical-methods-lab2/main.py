import sys
import numpy as np
import time
from datetime import datetime
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
    QTextEdit,
    QToolTip,
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class IntegralCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Численное интегрирование с графиком")
        self.setGeometry(100, 100, 1000, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0d1b2a;  /* dark-bg */
            }
            QWidget#inputPanel {
                background-color: #1b263b;  /* medium-bg */
                border-radius: 10px;
                padding: 15px;
            }
            QWidget#resultPanel {
                background-color: #1b263b;  /* medium-bg */
                border-radius: 10px;
                padding: 15px;
            }
            QLabel {
                font-size: 13px;  /* размер шрифта 13px */
                color: #ffffff;  /* белый текст */
            }
            QLineEdit {
                padding: 8px;
                font-size: 13px;  /* размер шрифта 13px */
                border: 1px solid #415a77;  /* primary-accent */
                border-radius: 5px;
                background-color: #1b263b;  /* medium-bg */
                color: #ffffff;  /* белый текст */
            }
            QLineEdit:focus {
                border: 1px solid #007ea7;  /* highlight */
            }
            QPushButton {
                padding: 10px;
                font-size: 13px;  /* размер шрифта 13px */
                background-color: #007ea7;  /* highlight */
                color: #ffffff;  /* белый текст */
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #006b8f;  /* немного темнее highlight */
            }
            QPushButton#clearButton {
                background-color: #cc7f15;  /* приглушённый оранжевый */
            }
            QPushButton#clearButton:hover {
                background-color: #b36f12;  /* ещё темнее */
            }
            QPushButton#saveButton {
                background-color: #4a9c8d;  /* graph-line-1 */
            }
            QPushButton#saveButton:hover {
                background-color: #3e8275;  /* немного темнее graph-line-1 */
            }
            QComboBox {
                padding: 8px;
                font-size: 13px;  /* размер шрифта 13px */
                border: 1px solid #415a77;  /* primary-accent */
                border-radius: 5px;
                background-color: #1b263b;  /* medium-bg */
                color: #ffffff;  /* белый текст */
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1b263b;  /* medium-bg */
                color: #ffffff;  /* белый текст */
                selection-background-color: #007ea7;  /* highlight */
                font-size: 13px;  /* размер шрифта 13px */
            }
            QTextEdit {
                font-size: 13px;  /* размер шрифта 13px */
                border: 1px solid #415a77;  /* primary-accent */
                border-radius: 5px;
                background-color: #1b263b;  /* medium-bg */
                color: #ffffff;  /* белый текст */
            }
        """
        )

        input_widget = QWidget()
        input_widget.setObjectName("inputPanel")
        input_layout = QVBoxLayout(input_widget)
        input_layout.setAlignment(Qt.AlignTop)
        input_layout.setSpacing(10)

        input_title = QLabel("Параметры интегрирования")
        input_title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #007ea7; margin-bottom: 10px;"
        )
        input_layout.addWidget(input_title)

        func_layout = QHBoxLayout()
        func_label = QLabel("Функция f(x):")
        self.func_input = QLineEdit("sin(x)")
        self.func_input.setToolTip(
            "Доступные функции: sin(x), cos(x), tan(x), exp(x), sqrt(x), abs(x), pi"
        )
        func_layout.addWidget(func_label)
        func_layout.addWidget(self.func_input)
        input_layout.addLayout(func_layout)

        func_hint = QLabel("(например, sin(x), cos(x), exp(x), sqrt(x), abs(x), pi)")
        func_hint.setStyleSheet("font-size: 11px; color: #778da9; margin-left: 5px;")
        input_layout.addWidget(func_hint)

        bounds_layout = QHBoxLayout()
        a_label = QLabel("Нижняя граница a:")
        self.a_input = QLineEdit("0")
        b_label = QLabel("Верхняя граница b:")
        self.b_input = QLineEdit("pi")
        bounds_layout.addWidget(a_label)
        bounds_layout.addWidget(self.a_input)
        bounds_layout.addWidget(b_label)
        bounds_layout.addWidget(self.b_input)
        input_layout.addLayout(bounds_layout)

        eps_layout = QHBoxLayout()
        eps_label = QLabel("Точность eps:")
        self.eps_input = QLineEdit("0.001")
        eps_layout.addWidget(eps_label)
        eps_layout.addWidget(self.eps_input)
        input_layout.addLayout(eps_layout)

        method_layout = QHBoxLayout()
        method_label = QLabel("Метод:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(
            ["Метод трапеций", "Метод Симпсона", "Метод Ньютона", "Все методы"]
        )
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        input_layout.addLayout(method_layout)

        buttons_layout = QHBoxLayout()
        self.calculate_button = QPushButton("Вычислить")
        self.calculate_button.clicked.connect(self.calculate_integral)
        self.clear_button = QPushButton("Очистить")
        self.clear_button.setObjectName("clearButton")
        self.clear_button.clicked.connect(self.clear_fields)
        self.save_button = QPushButton("Сохранить результаты")
        self.save_button.setObjectName("saveButton")
        self.save_button.clicked.connect(self.save_results)
        buttons_layout.addWidget(self.calculate_button)
        buttons_layout.addWidget(self.clear_button)
        buttons_layout.addWidget(self.save_button)
        input_layout.addLayout(buttons_layout)

        main_layout.addWidget(input_widget, 1)

        right_widget = QWidget()
        right_widget.setObjectName("resultPanel")
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)

        result_title = QLabel("Результаты и график")
        result_title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #007ea7; margin-bottom: 10px;"
        )
        right_layout.addWidget(result_title)

        self.figure = Figure()
        self.figure.patch.set_facecolor("#1b263b")
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        right_layout.addWidget(self.result_output)

        main_layout.addWidget(right_widget, 2)

        self.trap_points = None
        self.trap_values = None
        self.last_output = ""

    def f(self, x):
        """Функция для вычисления значения введенной функции"""
        try:
            return eval(
                self.func_input.text(),
                {
                    "x": x,
                    "sin": np.sin,
                    "cos": np.cos,
                    "tan": np.tan,
                    "exp": np.exp,
                    "sqrt": np.sqrt,
                    "abs": np.abs,
                    "pi": np.pi,
                },
            )
        except Exception as e:
            raise Exception(f"Ошибка в функции: {str(e)}")

    def trapezoidal_rule(self, a, b, eps):
        """Метод трапеций с оценкой погрешности по принципу Рунге"""
        start_time = time.time()
        n = 1
        h = (b - a) / n
        integral_prev = (self.f(a) + self.f(b)) * h / 2
        steps = []
        p = 2
        k = 2

        while True:
            n *= 2
            h = (b - a) / n
            integral = 0
            points = []
            values = []
            for i in range(n + 1):
                x = a + i * h
                points.append(x)
                values.append(self.f(x))
                if i == 0 or i == n:
                    integral += self.f(x) / 2
                else:
                    integral += self.f(x)
            integral *= h

            difference = abs(integral - integral_prev)
            runge_error = difference / (k**p - 1)
            steps.append(
                f"Итерация {len(steps) + 1}: Интеграл={integral:.6f}, Разница={difference:.6f}, Погрешность Рунге={runge_error:.6f}"
            )
            if runge_error < eps:
                end_time = time.time()
                self.trap_points = points
                self.trap_values = values
                return integral, steps, end_time - start_time, runge_error
            integral_prev = integral

    def simpson_rule(self, a, b, eps):
        """Метод Симпсона с оценкой погрешности по принципу Рунге"""
        start_time = time.time()
        n = 2
        h = (b - a) / n
        integral_prev = (self.f(a) + 4 * self.f((a + b) / 2) + self.f(b)) * h / 3
        steps = []
        p = 4
        k = 2

        while True:
            n *= 2
            h = (b - a) / n
            integral = self.f(a) + self.f(b)
            for i in range(1, n):
                x = a + i * h
                if i % 2 == 0:
                    integral += 2 * self.f(x)
                else:
                    integral += 4 * self.f(x)
            integral *= h / 3

            difference = abs(integral - integral_prev)
            runge_error = difference / (k**p - 1)
            steps.append(
                f"Итерация {len(steps) + 1}: Интеграл={integral:.6f}, Разница={difference:.6f}, Погрешность Рунге={runge_error:.6f}"
            )
            if runge_error < eps:
                end_time = time.time()
                return integral, steps, end_time - start_time, runge_error
            integral_prev = integral

    def newton_rule(self, a, b, eps):
        """Метод Ньютона с оценкой погрешности по принципу Рунге"""
        start_time = time.time()
        n = 3
        h = (b - a) / n
        integral_prev = 0
        for i in range(0, n, 3):
            x0 = a + i * h
            x1 = x0 + h
            x2 = x1 + h
            x3 = x2 + h
            integral_prev += (
                self.f(x0) + 3 * self.f(x1) + 3 * self.f(x2) + self.f(x3)
            ) * (3 * h / 8)
        steps = []
        p = 4
        k = 3

        while True:
            n *= 3
            h = (b - a) / n
            integral = 0
            for i in range(0, n, 3):
                x0 = a + i * h
                x1 = x0 + h
                x2 = x1 + h
                x3 = x2 + h
                integral += (
                    self.f(x0) + 3 * self.f(x1) + 3 * self.f(x2) + self.f(x3)
                ) * (3 * h / 8)

            difference = abs(integral - integral_prev)
            runge_error = difference / (k**p - 1)
            steps.append(
                f"Итерация {len(steps) + 1}: Интеграл={integral:.6f}, Разница={difference:.6f}, Погрешность Рунге={runge_error:.6f}"
            )
            if runge_error < eps:
                end_time = time.time()
                return integral, steps, end_time - start_time, runge_error
            integral_prev = integral

    def format_method_output(self, method_name, result, steps, exec_time, runge_error):
        """Форматирование вывода результатов для одного метода с использованием HTML"""
        output = f"{method_name}\n"
        output += f"Значение интеграла: {result:.6f}\n"
        output += f"Время работы: {exec_time:.6f} сек\n"
        output += f"Погрешность по Рунге: {runge_error:.6f}\n"
        output += "Промежуточные шаги:\n"
        output += "\n".join(steps)
        output += "\n" + "-" * 50 + "\n"
        return output

    def plot_function(self, a, b):
        """Построение графика функции, области интегрирования и трапеций"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#1b263b")
        ax.tick_params(colors="#ffffff", labelsize=13)
        ax.xaxis.label.set_color("#ffffff")
        ax.yaxis.label.set_color("#ffffff")
        ax.title.set_color("#ffffff")
        ax.title.set_size(15)
        ax.grid(color="#415a77")

        x = np.linspace(a - 1, b + 1, 1000)
        y = [self.f(xi) for xi in x]

        ax.plot(x, y, label=f"f(x) = {self.func_input.text()}", color="#4a9c8d")

        x_fill = np.linspace(a, b, 100)
        y_fill = [self.f(xi) for xi in x_fill]
        ax.fill_between(
            x_fill, y_fill, alpha=0.3, color="#ff9f1c", label="Область интегрирования"
        )

        if self.trap_points and self.trap_values:
            for i in range(len(self.trap_points) - 1):
                x_trap = [
                    self.trap_points[i],
                    self.trap_points[i],
                    self.trap_points[i + 1],
                    self.trap_points[i + 1],
                ]
                y_trap = [0, self.trap_values[i], self.trap_values[i + 1], 0]
                ax.fill(x_trap, y_trap, color="#415a77", alpha=0.2, edgecolor="#415a77")

        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title("График функции, область интегрирования и трапеции")
        ax.grid(True)
        ax.legend(
            facecolor="#1b263b", edgecolor="#415a77", labelcolor="#ffffff", fontsize=13
        )

        self.canvas.draw()

    def calculate_integral(self):
        """Основная функция для вычисления интеграла в зависимости от выбранного метода"""
        try:
            a = float(eval(self.a_input.text(), {"pi": np.pi}))
            b = float(eval(self.b_input.text(), {"pi": np.pi}))
            eps = float(self.eps_input.text())
            method = self.method_combo.currentText()

            if a >= b:
                self.result_output.setText(
                    "<span style='color: #ff9f1c;'>Ошибка: a должно быть меньше b</span>"
                )
                return

            self.trap_points = None
            self.trap_values = None

            output = f"Функция: {self.func_input.text()}\n"
            output += f"Границы: [{a}, {b}]\n"
            output += f"Точность: {eps}\n\n"

            if method == "Метод трапеций" or method == "Все методы":
                trap_result, trap_steps, trap_time, trap_runge_error = (
                    self.trapezoidal_rule(a, b, eps)
                )
                output += self.format_method_output(
                    "Метод трапеций",
                    trap_result,
                    trap_steps,
                    trap_time,
                    trap_runge_error,
                )

            if method == "Метод Симпсона" or method == "Все методы":
                simp_result, simp_steps, simp_time, simp_runge_error = (
                    self.simpson_rule(a, b, eps)
                )
                output += self.format_method_output(
                    "Метод Симпсона",
                    simp_result,
                    simp_steps,
                    simp_time,
                    simp_runge_error,
                )

            if method == "Метод Ньютона" or method == "Все методы":
                newt_result, newt_steps, newt_time, newt_runge_error = self.newton_rule(
                    a, b, eps
                )
                output += self.format_method_output(
                    "Метод Ньютона",
                    newt_result,
                    newt_steps,
                    newt_time,
                    newt_runge_error,
                )

            self.last_output = output
            html_output = output.replace("\n", "<br>").replace(
                "Значение интеграла",
                "<span style='color: #ff9f1c;'>Значение интеграла</span>",
            )
            self.result_output.setHtml(html_output)

            self.plot_function(a, b)

        except ValueError as e:
            self.result_output.setHtml(
                f"<span style='color: #ff9f1c;'>Ошибка: Введите корректные числовые значения<br>{str(e)}</span>"
            )
        except Exception as e:
            self.result_output.setHtml(
                f"<span style='color: #ff9f1c;'>Ошибка: {str(e)}</span>"
            )

    def save_results(self):
        """Сохранение результатов в текстовый файл"""
        if not self.last_output:
            self.result_output.setHtml(
                "<span style='color: #ff9f1c;'>Ошибка: Нет результатов для сохранения</span>"
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"integral_results_{timestamp}.txt"

        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(self.last_output)
            self.result_output.setHtml(
                f"<span style='color: #4a9c8d;'>Результаты сохранены в файл: {filename}</span>"
            )
        except Exception as e:
            self.result_output.setHtml(
                f"<span style='color: #ff9f1c;'>Ошибка при сохранении файла: {str(e)}</span>"
            )

    def clear_fields(self):
        """Очистка всех полей ввода и вывода"""
        self.func_input.setText("sin(x)")
        self.a_input.setText("0")
        self.b_input.setText("pi")
        self.eps_input.setText("0.001")
        self.method_combo.setCurrentIndex(0)
        self.result_output.clear()
        self.last_output = ""
        self.trap_points = None
        self.trap_values = None
        self.figure.clear()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IntegralCalculator()
    window.show()
    sys.exit(app.exec_())
