import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QLabel,
    QRadioButton,
    QGroupBox,
    QFileDialog,
    QSplitter,
)
from PyQt5.QtCore import Qt, QLocale
from PyQt5.QtGui import QFont, QDoubleValidator
from scipy.interpolate import CubicSpline


class NumericalDiffWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Численное дифференцирование")
        self.setGeometry(100, 100, 900, 650)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setContentsMargins(10, 10, 10, 10)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_widget.setLayout(left_layout)

        font = QFont("Arial", 10)
        left_widget.setFont(font)

        self.function_group = QGroupBox("Способ задания функции:")
        self.function_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; padding: 10px; }"
        )
        function_layout = QHBoxLayout()
        function_layout.setSpacing(10)
        self.analytical_radio = QRadioButton("Аналитически")
        self.table_radio = QRadioButton("Таблично")
        self.analytical_radio.setChecked(True)
        function_layout.addWidget(self.analytical_radio)
        function_layout.addWidget(self.table_radio)
        self.function_group.setLayout(function_layout)
        left_layout.addWidget(self.function_group)

        self.function_input = QLineEdit()
        self.function_input.setPlaceholderText("x**2")
        self.function_input.setFixedHeight(35)
        self.function_input.setToolTip(
            "Введите функцию, например: x**2, sin(x), exp(x)"
        )
        left_layout.addWidget(QLabel("Функция f(x):"))
        left_layout.addWidget(self.function_input)

        self.available_funcs_label = QLabel(
            "Доступные функции: sin, cos, tan, exp, sqrt, abs"
        )
        self.available_funcs_label.setWordWrap(True)
        self.available_funcs_label.setStyleSheet("color: #555; font-style: italic;")
        left_layout.addWidget(self.available_funcs_label)

        self.table_input_label = QLabel(
            "Введите данные (x y, каждая строка — пара значений):"
        )
        self.table_input_label.setVisible(False)
        left_layout.addWidget(self.table_input_label)

        self.table_input = QTextEdit()
        self.table_input.setFixedHeight(100)
        self.table_input.setPlaceholderText("Пример:\n0.6 0.422\n0.7 0.562\n0.8 0.722")
        self.table_input.setVisible(False)
        left_layout.addWidget(self.table_input)

        self.confirm_table_btn = QPushButton("Подтвердить данные")
        self.confirm_table_btn.setFixedHeight(40)
        self.confirm_table_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; border-radius: 5px; } QPushButton:hover { background-color: #1976D2; }"
        )
        self.confirm_table_btn.clicked.connect(self.confirm_table_data)
        self.confirm_table_btn.setVisible(False)
        left_layout.addWidget(self.confirm_table_btn)

        self.load_file_btn = QPushButton("Загрузить из файла")
        self.load_file_btn.setFixedHeight(40)
        self.load_file_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; border-radius: 5px; } QPushButton:hover { background-color: #F57C00; }"
        )
        self.load_file_btn.clicked.connect(self.load_table_from_file)
        self.load_file_btn.setVisible(False)
        left_layout.addWidget(self.load_file_btn)

        self.x0_input = QLineEdit()
        self.x0_input.setPlaceholderText("1.0")
        self.x0_input.setFixedHeight(35)
        validator_x0 = QDoubleValidator()
        validator_x0.setLocale(QLocale("C"))
        self.x0_input.setValidator(validator_x0)
        self.x0_input.setToolTip("Введите точку x0, например: 1.0")
        left_layout.addWidget(QLabel("Точка x:"))
        left_layout.addWidget(self.x0_input)

        self.order_group = QGroupBox("Порядок производной:")
        self.order_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; padding: 10px; }"
        )
        order_layout = QHBoxLayout()
        order_layout.setSpacing(10)
        self.first_order_radio = QRadioButton("Первая производная")
        self.second_order_radio = QRadioButton("Вторая производная")
        self.first_order_radio.setChecked(True)
        order_layout.addWidget(self.first_order_radio)
        order_layout.addWidget(self.second_order_radio)
        self.order_group.setLayout(order_layout)
        left_layout.addWidget(self.order_group)

        self.calculate_btn = QPushButton("Вычислить производную")
        self.calculate_btn.setFixedHeight(40)
        self.calculate_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; border-radius: 5px; } QPushButton:hover { background-color: #45a049; }"
        )
        self.calculate_btn.clicked.connect(self.calculate_derivative)
        left_layout.addWidget(self.calculate_btn)

        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.setFixedHeight(40)
        self.clear_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-weight: bold; border-radius: 5px; } QPushButton:hover { background-color: #da190b; }"
        )
        self.clear_btn.clicked.connect(self.clear_fields)
        left_layout.addWidget(self.clear_btn)

        left_layout.addStretch(1)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_widget.setLayout(right_layout)

        result_label = QLabel("Результаты дифференцирования:")
        result_label.setFont(QFont("Arial", 11, QFont.Bold))
        right_layout.addWidget(result_label)

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        self.result_output.setFont(QFont("Courier New", 10))
        self.result_output.setStyleSheet(
            "border: 1px solid gray; border-radius: 5px; padding: 5px;"
        )
        right_layout.addWidget(self.result_output)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        splitter.setSizes([350, 550])

        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_widget.setLayout(main_layout)

        self.table_data = None

        self.table_radio.toggled.connect(self.toggle_function_input)

        self.eval_locals = {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "x": None,
        }

    def toggle_function_input(self, checked):
        if checked:
            self.function_input.setEnabled(False)
            self.available_funcs_label.setVisible(False)
            self.table_input_label.setVisible(True)
            self.table_input.setVisible(True)
            self.confirm_table_btn.setVisible(True)
            self.load_file_btn.setVisible(True)
            self.table_data = None
        else:
            self.function_input.setEnabled(True)
            self.available_funcs_label.setVisible(True)
            self.table_input_label.setVisible(False)
            self.table_input.setVisible(False)
            self.confirm_table_btn.setVisible(False)
            self.load_file_btn.setVisible(False)
            self.table_data = None

    def clear_fields(self):
        self.function_input.clear()
        self.x0_input.clear()
        self.table_input.clear()
        self.result_output.clear()
        self.analytical_radio.setChecked(True)
        self.first_order_radio.setChecked(True)
        self.table_data = None

    def load_table_from_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Открыть файл таблицы", "", "Text Files (*.txt)"
        )
        if file_name:
            try:
                self.table_data = np.loadtxt(file_name)
                self.result_output.setText("Данные успешно загружены из файла.\n")
                return True
            except Exception as e:
                self.result_output.setText(
                    f"Ошибка загрузки таблицы из файла: {str(e)}\n"
                )
                self.table_data = None
                return False
        return False

    def confirm_table_data(self):
        try:
            text = self.table_input.toPlainText().strip()
            if not text:
                self.result_output.setText("Ошибка: введите данные!\n")
                self.table_data = None
                return False

            lines = text.split("\n")
            data = []
            for line in lines:
                line = line.strip()
                if line:
                    values = line.split()
                    if len(values) != 2:
                        raise ValueError(
                            "Каждая строка должна содержать ровно два значения (x y)."
                        )
                    x, y = float(values[0]), float(values[1])
                    data.append([x, y])
            self.table_data = np.array(data)
            self.result_output.setText("Данные успешно введены.\n")
            return True
        except Exception as e:
            self.result_output.setText(f"Ошибка ввода данных: {str(e)}\n")
            self.table_data = None
            return False

    def numerical_derivative(self, x, y, x0, h, order, derivative_order=1):
        idx = np.argmin(np.abs(x - x0))
        if not np.isclose(x[idx], x0, rtol=1e-5):
            raise ValueError("Точка x0 не совпадает с узлом сетки!")

        if derivative_order == 1:
            if idx == 0:
                return (-3 * y[idx] + 4 * y[idx + 1] - y[idx + 2]) / (2 * h)
            elif idx == len(x) - 1:
                return (3 * y[idx] - 4 * y[idx - 1] + y[idx - 2]) / (2 * h)
            else:
                return (y[idx + 1] - y[idx - 1]) / (2 * h)
        elif derivative_order == 2:
            if idx == 0:
                return (y[idx] - 2 * y[idx + 1] + y[idx + 2]) / (h**2)
            elif idx == len(x) - 1:
                return (y[idx] - 2 * y[idx - 1] + y[idx - 2]) / (h**2)
            else:
                return (y[idx + 1] - 2 * y[idx] + y[idx - 1]) / (h**2)
        elif derivative_order == 3:
            if idx < 2 or idx > len(x) - 3:
                raise ValueError(
                    "Недостаточно точек для вычисления третьей производной"
                )
            return (y[idx + 2] - 2 * y[idx + 1] + 2 * y[idx - 1] - y[idx - 2]) / (
                2 * h**3
            )
        elif derivative_order == 4:
            if idx < 2 or idx > len(x) - 3:
                raise ValueError(
                    "Недостаточно точек для вычисления четвёртой производной"
                )
            return (
                y[idx + 2] - 4 * y[idx + 1] + 6 * y[idx] - 4 * y[idx - 1] + y[idx - 2]
            ) / (h**4)
        else:
            raise ValueError("Неподдерживаемый порядок производной")

    def estimate_M(self, x0, h, order, func_str=None, x=None, y=None):
        try:
            if func_str:
                x_vals = np.arange(x0 - h * 4, x0 + h * 4 + h, h)
                self.eval_locals["x"] = x_vals
                y_vals = eval(func_str, {"__builtins__": None}, self.eval_locals)
            else:
                x_vals = x
                y_vals = y

            if order == 1:
                f3 = self.numerical_derivative(
                    x_vals, y_vals, x0, h, order, derivative_order=3
                )
                return abs(f3)
            else:
                f4 = self.numerical_derivative(
                    x_vals, y_vals, x0, h, order, derivative_order=4
                )
                return abs(f4)
        except Exception as e:
            raise ValueError(f"Не удалось оценить M: {str(e)}")

    def optimal_h(self, order, x0, func_str=None, x=None, y=None, eps=1e-10):
        h_initial = 1e-3
        if func_str:
            M = self.estimate_M(x0, h_initial, order, func_str=func_str)
        else:
            x_vals = np.arange(
                x0 - h_initial * 4, x0 + h_initial * 4 + h_initial, h_initial
            )
            cs = CubicSpline(x, y)
            y_vals = cs(x_vals)
            M = self.estimate_M(x0, h_initial, order, x=x_vals, y=y_vals)

        if M == 0:
            return 1e-6

        if order == 1:
            h = (3 * eps / M) ** (1 / 3)
        else:
            h = 2 * (3 * eps / M) ** (1 / 4)

        return max(h, 1e-6)

    def runge_error(self, x0, h, order, func_str=None, x=None, y=None):
        if func_str:
            x_h = np.arange(x0 - h * 4, x0 + h * 4 + h, h)
            idx_h = np.argmin(np.abs(x_h - x0))
            if not np.isclose(x_h[idx_h], x0, rtol=1e-5):
                raise ValueError("Точка x0 не совпадает с узлом сетки для шага h")
            self.eval_locals["x"] = x_h
            y_h = eval(func_str, {"__builtins__": None}, self.eval_locals)
            d1 = self.numerical_derivative(
                x_h, y_h, x0, h, order, derivative_order=order
            )

            x_2h = np.arange(x0 - 2 * h * 4, x0 + 2 * h * 4 + 2 * h, 2 * h)
            idx_2h = np.argmin(np.abs(x_2h - x0))
            if not np.isclose(x_2h[idx_2h], x0, rtol=1e-5):
                raise ValueError("Точка x0 не совпадает с узлом сетки для шага 2h")
            self.eval_locals["x"] = x_2h
            y_2h = eval(func_str, {"__builtins__": None}, self.eval_locals)
            d2 = self.numerical_derivative(
                x_2h, y_2h, x0, 2 * h, order, derivative_order=order
            )
        else:
            d1 = self.numerical_derivative(x, y, x0, h, order, derivative_order=order)

            idx = np.argmin(np.abs(x - x0))
            if idx < 1 or idx > len(x) - 2:
                raise ValueError(
                    "Недостаточно точек для вычисления производной с шагом h"
                )

            x_2h = np.arange(x0 - 2 * h * 4, x0 + 2 * h * 4 + 2 * h, 2 * h)
            idx_2h = np.argmin(np.abs(x_2h - x0))
            if not np.isclose(x_2h[idx_2h], x0, rtol=1e-5):
                raise ValueError("Точка x0 не совпадает с узлом сетки для шага 2h")
            if idx_2h < 1 or idx_2h > len(x_2h) - 2:
                raise ValueError(
                    "Недостаточно точек для вычисления производной с шагом 2h"
                )

            cs = CubicSpline(x, y)
            y_2h = cs(x_2h)

            d2 = self.numerical_derivative(
                x_2h, y_2h, x0, 2 * h, order, derivative_order=order
            )

        return abs(d1 - d2) / (2**order - 1)

    def residual_error(self, x, y, x0, h, order):
        try:
            if order == 1:
                f3 = self.numerical_derivative(x, y, x0, h, order, derivative_order=3)
                return (h**2) / 6 * abs(f3)
            else:
                f4 = self.numerical_derivative(x, y, x0, h, order, derivative_order=4)
                return (h**2) / 12 * abs(f4)
        except Exception as e:
            return None

    def computational_error(self, runge_err, derivative):
        machine_eps = np.finfo(float).eps
        machine_err = abs(derivative) * machine_eps
        return runge_err + machine_err

    def calculate_derivative(self):
        self.result_output.clear()

        try:
            if not self.x0_input.text().strip():
                self.result_output.setText("Ошибка: укажите точку x0!\n")
                return
            x0 = float(self.x0_input.text())

            order = 1 if self.first_order_radio.isChecked() else 2

            machine_eps = np.finfo(float).eps
            E = machine_eps / 2

            if self.analytical_radio.isChecked():
                result = "1. ВХОДНЫЕ ДАННЫЕ:\n"
                result += f"Способ задания: Аналитически\n"
                result += f"x0 = {x0:.6g}\n"
                result += f"Порядок производной: {order}\n"

                func_str = self.function_input.text()
                if not func_str:
                    self.result_output.setText("Ошибка: укажите функцию!\n")
                    return
                result += f"Функция: {func_str}\n"

                h = self.optimal_h(order, x0, func_str=func_str, eps=E)
                result += f"Автоматический шаг h: {h:.6g}\n"
                result += "\n"

                try:
                    x = np.arange(x0 - h * 4, x0 + h * 4 + h, h)
                    idx = np.argmin(np.abs(x - x0))
                    if not np.isclose(x[idx], x0, rtol=1e-5):
                        self.result_output.setText(
                            "Ошибка: точка x0 не совпадает с узлом сетки!\n"
                        )
                        return

                    self.eval_locals["x"] = x
                    y = eval(func_str, {"__builtins__": None}, self.eval_locals)

                    result += "2. ТАБЛИЦА ЗНАЧЕНИЙ ФУНКЦИИ:\n"
                    result += f"x: [{', '.join([f'{xi:.6g}' for xi in x])}]\n"
                    result += f"f(x): [{', '.join([f'{yi:.6g}' for yi in y])}]\n"
                    result += "\n"

                    derivative = self.numerical_derivative(
                        x, y, x0, h, order, derivative_order=order
                    )
                    runge_err = self.runge_error(x0, h, order, func_str=func_str)
                    residual_err = self.residual_error(x, y, x0, h, order)
                    comp_err = self.computational_error(runge_err, derivative)

                    result += "3. ИТОГОВЫЙ РЕЗУЛЬТАТ:\n"
                    result += f"Значение производной: {derivative:.20f}\n"
                    result += f"Погрешность по Рунге: {runge_err:.20f}\n"
                    if residual_err is not None:
                        result += f"Остаточный член: {residual_err:.20f}\n"
                    else:
                        result += "Остаточный член: не удалось вычислить (недостаточно точек для производной высокого порядка)\n"
                    result += f"Вычислительная погрешность: {comp_err:.20f}\n"
                    result += "\n"

                except Exception as e:
                    result += f"Ошибка вычисления: {str(e)}\n"

            else:
                if self.table_data is None:
                    self.result_output.setText(
                        "Ошибка: таблица не загружена! Введите данные или загрузите файл.\n"
                    )
                    return

                x_orig = self.table_data[:, 0]
                y_orig = self.table_data[:, 1]

                if not np.all(np.diff(x_orig) > 0):
                    self.result_output.setText(
                        "Ошибка: значения x в таблице должны быть упорядочены по возрастанию!\n"
                    )
                    return

                h_actual = x_orig[1] - x_orig[0]
                if not np.allclose(np.diff(x_orig), h_actual):
                    self.result_output.setText(
                        "Ошибка: сетка должна быть равномерной!\n"
                    )
                    return

                h = self.optimal_h(order, x0, x=x_orig, y=y_orig, eps=E)
                h = max(h, h_actual)
                h = h_actual * round(h / h_actual)
                result = "1. ВХОДНЫЕ ДАННЫЕ:\n"
                result += f"Способ задания: Таблично\n"
                result += f"x0 = {x0:.6g}\n"
                result += f"Порядок производной: {order}\n"
                result += f"Автоматический шаг h: {h:.6g} (скорректированный под сетку таблицы)\n"
                result += "\n"

                x = np.arange(x0 - h * 4, x0 + h * 4 + h, h)
                idx = np.argmin(np.abs(x - x0))
                if not np.isclose(x[idx], x0, rtol=1e-5):
                    self.result_output.setText(
                        "Ошибка: точка x0 не совпадает с узлом новой сетки!\n"
                    )
                    return

                cs = CubicSpline(x_orig, y_orig)
                y = cs(x)

                result += "2. ТАБЛИЦА ЗНАЧЕНИЙ ФУНКЦИИ:\n"
                result += f"x: [{', '.join([f'{xi:.6g}' for xi in x])}]\n"
                result += f"f(x): [{', '.join([f'{yi:.6g}' for yi in y])}]\n"
                result += "\n"

                derivative = self.numerical_derivative(
                    x, y, x0, h, order, derivative_order=order
                )
                runge_err = self.runge_error(x0, h, order, x=x, y=y)
                residual_err = self.residual_error(x, y, x0, h, order)
                comp_err = self.computational_error(runge_err, derivative)

                result += "3. ИТОГОВЫЙ РЕЗУЛЬТАТ:\n"
                result += f"Значение производной: {derivative:.20f}\n"
                result += f"Погрешность по Рунге: {runge_err:.20f}\n"
                if residual_err is not None:
                    result += f"Остаточный член: {residual_err:.20f}\n"
                else:
                    result += "Остаточный член: не удалось вычислить (недостаточно точек для производной высокого порядка)\n"
                result += f"Вычислительная погрешность: {comp_err:.20f}\n"
                result += "\n"

            self.result_output.setText(result)

        except ValueError as e:
            self.result_output.setText(f"Ошибка: {str(e)}\n")
        except Exception as e:
            self.result_output.setText(f"Непредвиденная ошибка: {str(e)}\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NumericalDiffWindow()
    window.show()
    sys.exit(app.exec_())
