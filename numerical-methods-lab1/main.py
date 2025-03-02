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
    QLineEdit,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QTextEdit,
)
from PyQt5.QtCore import Qt


def lagrange_interpolation(x, points):
    """Вычисляет интерполяцию Лагранжа для заданной точки x."""
    n = len(points)
    result = 0.0
    for i in range(n):
        xi, yi = points[i]
        term = yi
        for j in range(n):
            if i != j:
                xj, _ = points[j]
                term *= (x - xj) / (xi - xj)
        result += term
    return result


def newton_interpolation(x, points):
    """Вычисляет интерполяцию Ньютона для заданной точки x."""
    n = len(points)
    # Матрица разделённых разностей
    divided_diff = [[0] * n for _ in range(n)]

    # Первая колонка (значения функции в узлах)
    for i in range(n):
        divided_diff[i][0] = points[i][1]

    # Заполнение таблицы разделённых разностей
    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (
                divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]
            ) / (points[i + j][0] - points[i][0])

    # Формирование итогового полинома
    result = divided_diff[0][0]
    product = 1.0
    for i in range(1, n):
        product *= x - points[i - 1][0]
        result += divided_diff[0][i] * product
    return result


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Интерполяция многочленов")
        self.setGeometry(100, 100, 1100, 800)

        self.colors = {
            "dark-bg": "#0d1b2a",  # Очень тёмный синий
            "medium-bg": "#1b263b",  # Тёмный сине-серый
            "primary-accent": "#415a77",  # Средний синий
            "secondary-accent": "#778da9",  # Светлый синий
            "text-primary": "#e0e1dd",  # Светло-серый
            "highlight": "#007ea7",  # Церулеан
            "graph-line-1": "#4a9c8d",  # Бирюзовый
            "graph-line-2": "#ff9f1c",  # Оранжевый
        }

        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: {self.colors["dark-bg"]};
            }}
            QLabel {{
                font-size: 14px;
                font-weight: bold;
                color: {self.colors["text-primary"]};
                margin-bottom: 5px;
            }}
            QLineEdit {{
                padding: 10px;
                border: 2px solid {self.colors["primary-accent"]};
                border-radius: 6px;
                font-size: 14px;
                background-color: {self.colors["medium-bg"]};
                color: {self.colors["text-primary"]};
            }}
            QPushButton {{
                background-color: {self.colors["primary-accent"]};
                color: {self.colors["text-primary"]};
                border: none;
                padding: 12px;
                font-size: 14px;
                border-radius: 6px;
                margin-top: 10px;
            }}
            QPushButton:hover {{
                background-color: {self.colors["secondary-accent"]};
            }}
            QPushButton:pressed {{
                background-color: {self.colors["highlight"]};
            }}
            QTextEdit {{
                font-size: 14px;
                color: {self.colors["text-primary"]};
                padding: 15px;
                background-color: {self.colors["medium-bg"]};
                border: 1px solid {self.colors["primary-accent"]};
                border-radius: 6px;
            }}
        """
        )

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()

        left_panel = QVBoxLayout()
        left_panel.setSpacing(20)
        left_panel.setContentsMargins(20, 20, 20, 20)

        left_panel.addWidget(QLabel("Ввод данных"))

        self.points_input = QLineEdit()
        self.points_input.setPlaceholderText("Пример: 1,2 2,3 3,4")
        left_panel.addWidget(QLabel("Узловые точки:"))
        left_panel.addWidget(self.points_input)

        self.x_input = QLineEdit()
        self.x_input.setPlaceholderText("Пример: 2.5")
        left_panel.addWidget(QLabel("Точка x*:"))
        left_panel.addWidget(self.x_input)

        self.file_button = QPushButton("Загрузить из файла")
        self.file_button.clicked.connect(self.input_from_file)
        left_panel.addWidget(self.file_button)

        self.calculate_button = QPushButton("Рассчитать")
        self.calculate_button.clicked.connect(self.calculate)
        left_panel.addWidget(self.calculate_button)

        self.save_text_button = QPushButton("Сохранить текст")
        self.save_text_button.clicked.connect(self.save_text_results)
        left_panel.addWidget(self.save_text_button)

        self.save_graph_button = QPushButton("Сохранить график")
        self.save_graph_button.clicked.connect(self.save_graph)
        left_panel.addWidget(self.save_graph_button)

        left_panel.addStretch()

        right_panel = QVBoxLayout()
        right_panel.setSpacing(20)
        right_panel.setContentsMargins(20, 20, 20, 20)

        right_panel.addWidget(QLabel("Результаты и график"))

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        right_panel.addWidget(self.result_text)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        right_panel.addWidget(self.canvas)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 3)
        central_widget.setLayout(main_layout)

    def input_from_file(self):
        """Загружает данные из выбранного файла."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", "", "Текстовые файлы (*.txt)"
        )
        if file_name:
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    self.points_input.setText(lines[0].strip())
                    self.x_input.setText(lines[1].strip())
                    self.calculate()
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка", f"Не удалось загрузить файл: {str(e)}"
                )

    def get_points(self):
        """Преобразует строку с точками в список кортежей (x, y)."""
        points_str = self.points_input.text().strip().split()
        points = []
        for p in points_str:
            x, y = map(float, p.split(","))
            points.append((x, y))
        return points

    def calculate(self):
        """Выполняет расчёты (Лагранж, Ньютон) и выводит результаты."""
        try:
            points = self.get_points()
            x_star = float(self.x_input.text())

            lagrange_result = lagrange_interpolation(x_star, points)
            newton_result = newton_interpolation(x_star, points)

            result_text = f"Узловые точки: {points}\nТочка x*: {x_star}\n\n"

            result_text += "--- Многочлен Лагранжа ---\nБазисные многочлены:\n"
            for i, (xi, yi) in enumerate(points):
                term = 1.0
                term_str = f"l_{i}({x_star}) = "
                for j, (xj, _) in enumerate(points):
                    if i != j:
                        term *= (x_star - xj) / (xi - xj)
                        term_str += f"({x_star} - {xj})/({xi} - {xj}) * "
                term_str = term_str.rstrip(" * ") + f" = {term:.4f}\n"
                result_text += term_str
            result_text += f"Значение: L({x_star}) = {lagrange_result:.4f}\n\n"

            result_text += "--- Многочлен Ньютона ---\nРазделённые разности:\n"
            n = len(points)
            dd = [[0] * n for _ in range(n)]

            for i in range(n):
                dd[i][0] = points[i][1]
                result_text += f"f[x_{i}] = {dd[i][0]}\n"

            for j in range(1, n):
                for i in range(n - j):
                    dd[i][j] = (dd[i + 1][j - 1] - dd[i][j - 1]) / (
                        points[i + j][0] - points[i][0]
                    )
                    result_text += (
                        f"f[x_{i}, x_{i+j}] = "
                        f"({dd[i + 1][j - 1]} - {dd[i][j - 1]})"
                        f"/({points[i + j][0]} - {points[i][0]}) = {dd[i][j]:.4f}\n"
                    )
            result_text += f"Значение: P({x_star}) = {newton_result:.4f}\n\n"

            diff = abs(lagrange_result - newton_result)
            result_text += f"Разница между результатами: |{lagrange_result:.4f} - {newton_result:.4f}| = {diff:.4f}"
            self.result_text.setText(result_text)

            self.plot_interpolation(points, x_star)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка ввода: {str(e)}")

    def plot_interpolation(self, points, x_star):
        """Строит график интерполяционных многочленов Лагранжа и Ньютона."""
        x_vals = np.linspace(
            min([p[0] for p in points]) - 1, max([p[0] for p in points]) + 1, 100
        )

        self.ax.clear()

        plt.style.use("dark_background")
        self.ax.set_facecolor(self.colors["dark-bg"])
        self.fig.patch.set_facecolor(self.colors["dark-bg"])

        self.ax.plot(
            x_vals,
            [lagrange_interpolation(x, points) for x in x_vals],
            color=self.colors["graph-line-1"],
            linestyle="-",
            linewidth=2,
            label="Многочлен Лагранжа",
        )
        self.ax.plot(
            x_vals,
            [newton_interpolation(x, points) for x in x_vals],
            color=self.colors["graph-line-2"],
            linestyle="--",
            linewidth=2,
            label="Многочлен Ньютона",
        )

        self.ax.scatter(
            [p[0] for p in points],
            [p[1] for p in points],
            color=self.colors["secondary-accent"],
            s=50,
            zorder=5,
            label="Узлы интерполяции",
            edgecolor="w",
        )

        self.ax.scatter(
            [x_star],
            [lagrange_interpolation(x_star, points)],
            color=self.colors["graph-line-1"],
            s=70,
            zorder=6,
            label=f"x*={x_star} (Лагранж)",
            edgecolor="w",
        )
        self.ax.scatter(
            [x_star],
            [newton_interpolation(x_star, points)],
            color=self.colors["graph-line-2"],
            s=70,
            zorder=6,
            label=f"x*={x_star} (Ньютон)",
            edgecolor="w",
        )

        self.ax.set_title(
            "Интерполяционные многочлены",
            fontsize=16,
            pad=15,
            color=self.colors["text-primary"],
        )
        self.ax.set_xlabel("x", fontsize=12, color=self.colors["text-primary"])
        self.ax.set_ylabel("y", fontsize=12, color=self.colors["text-primary"])

        self.ax.grid(True, linestyle="--", alpha=0.3, color=self.colors["text-primary"])

        legend = self.ax.legend(
            loc="upper left",
            fontsize=12,
            facecolor=self.colors["medium-bg"] + "CC",
            edgecolor=self.colors["primary-accent"],
        )
        for text in legend.get_texts():
            text.set_color(self.colors["text-primary"])

        self.canvas.draw()

    def save_text_results(self):
        """Сохраняет текстовые результаты в выбранный файл."""
        try:
            text_file, _ = QFileDialog.getSaveFileName(
                self, "Сохранить текст", "", "Текстовые файлы (*.txt)"
            )
            if text_file:
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(self.result_text.toPlainText())
                QMessageBox.information(self, "Успех", "Текст сохранён!")
        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка", f"Не удалось сохранить текст: {str(e)}"
            )

    def save_graph(self):
        """Сохраняет график в файл (формат PNG)."""
        try:
            graph_file, _ = QFileDialog.getSaveFileName(
                self, "Сохранить график", "", "Изображения (*.png)"
            )
            if graph_file:
                self.fig.savefig(graph_file, dpi=300, bbox_inches="tight")
                QMessageBox.information(self, "Успех", "График сохранён!")
        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка", f"Не удалось сохранить график: {str(e)}"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
