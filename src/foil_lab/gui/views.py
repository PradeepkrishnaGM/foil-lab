# src/foil_lab/gui/views.py
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class AirfoilPlotCanvas(QWidget):
    """A custom PySide6 widget that houses a Matplotlib figure."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent)
        
        # Create the matplotlib Figure and Canvas
        self.fig = Figure(figsize=(width, height), dpi=dpi, layout="constrained")   # re-laid out on every draw
        self.canvas = FigureCanvas(self.fig)
        
        # Create the standard Matplotlib toolbar (zoom, pan, save)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Layout the toolbar and canvas vertically
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Create default axes
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Foil Lab Workspace")
        self.ax.set_xlabel("x/c")
        self.ax.set_ylabel("y/c")
        self.ax.axis("equal")
        self.ax.grid(True, linestyle=':', alpha=0.7)
        
    def clear_plot(self):
        """Clears the axes for a new plot."""
        self.ax.clear()
        self.ax.grid(True, linestyle=':', alpha=0.7)
        
    def redraw(self):
        """Redraw when the canvas is next painted (a hidden tab is drawn when shown, at its real
        size)."""
        self.canvas.draw_idle()


class StatusBox(QPlainTextEdit):
    """Read-only, selectable, scrollable status text with the ``setText`` / ``text`` interface of a
    label, so long results (drag parts, corrections, warnings) are never cut off."""

    def __init__(self, text: str = "", parent=None, lines: int = 9):
        super().__init__(text, parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.setFixedHeight(int(self.fontMetrics().lineSpacing() * lines + 12))

    def setText(self, text: str) -> None:
        self.setPlainText(text)

    def text(self) -> str:
        return self.toPlainText()
