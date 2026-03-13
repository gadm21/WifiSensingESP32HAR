"""
Shared tabbed GUI for displaying multiple matplotlib figures with per-plot export buttons.

Usage
-----
    from vis_gui import PlotGUI

    gui = PlotGUI("My Results")
    gui.add_plot("Accuracy Bars", fig1, "plot_accuracy_bars.png")
    gui.add_plot("Heatmap", fig2, "plot_heatmap.png")
    gui.show()
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.backends.backend_tkagg import (          # noqa: E402
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)


class PlotGUI:
    """Single-window, tabbed matplotlib viewer with per-plot export."""

    # ── dark-ish theme colours ──────────────────────────────────────
    BG        = '#1e1e2e'
    FG        = '#cdd6f4'
    TAB_BG    = '#313244'
    BTN_BG    = '#45475a'
    BTN_FG    = '#cdd6f4'
    ACCENT    = '#89b4fa'

    def __init__(self, title="Visualization Suite", results_dir=None):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg=self.BG)
        try:
            self.root.state('zoomed')          # Windows maximize
        except tk.TclError:
            self.root.attributes('-zoomed', True)  # Linux fallback

        self.results_dir = results_dir or os.getcwd()
        self._figures = []   # list of (tab_name, fig, default_filename)

        # ── ttk style ───────────────────────────────────────────────
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('TNotebook',      background=self.BG)
        style.configure('TNotebook.Tab',  background=self.TAB_BG,
                        foreground=self.FG, padding=[12, 4])
        style.map('TNotebook.Tab',
                  background=[('selected', self.ACCENT)],
                  foreground=[('selected', '#1e1e2e')])
        style.configure('TFrame',   background=self.BG)
        style.configure('TLabel',   background=self.BG, foreground=self.FG)
        style.configure('Export.TButton', background=self.BTN_BG,
                        foreground=self.BTN_FG, padding=[8, 3])

        # ── top bar ────────────────────────────────────────────────
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=6, pady=(6, 2))

        ttk.Label(top, text=title,
                  font=('Segoe UI', 14, 'bold')).pack(side=tk.LEFT)

        btn_all = ttk.Button(top, text='Export All PNGs',
                             style='Export.TButton',
                             command=self._export_all)
        btn_all.pack(side=tk.RIGHT, padx=4)

        # ── notebook (tabs) ────────────────────────────────────────
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

    # ── public API ──────────────────────────────────────────────────
    def add_plot(self, name, fig, default_filename=None):
        """Embed *fig* as a new tab called *name*."""
        if default_filename is None:
            safe = name.lower().replace(' ', '_').replace(':', '')
            safe = safe.replace('/', '_').replace('\\', '_')
            default_filename = safe + '.png'

        self._figures.append((name, fig, default_filename))

        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=name)

        # button bar
        bar = ttk.Frame(tab)
        bar.pack(fill=tk.X, padx=4, pady=2)

        def _export(f=fig, fn=default_filename):
            path = filedialog.asksaveasfilename(
                initialdir=self.results_dir,
                initialfile=fn,
                defaultextension='.png',
                filetypes=[('PNG', '*.png'), ('SVG', '*.svg'),
                           ('PDF', '*.pdf'), ('All files', '*.*')])
            if path:
                f.savefig(path, dpi=150, bbox_inches='tight')
                print(f"[export] Saved {path}")

        ttk.Button(bar, text=f"Export \"{name}\"",
                   style='Export.TButton',
                   command=_export).pack(side=tk.RIGHT)

        # matplotlib canvas
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)

        # navigation toolbar (zoom / pan / home)
        toolbar_frame = ttk.Frame(tab)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(canvas, toolbar_frame)

    def show(self):
        """Enter the Tk main-loop (blocking)."""
        self.root.mainloop()

    # ── internal ────────────────────────────────────────────────────
    def _export_all(self):
        directory = filedialog.askdirectory(initialdir=self.results_dir)
        if not directory:
            return
        for name, fig, fname in self._figures:
            path = os.path.join(directory, fname)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"[export] Saved {path}")
        print(f"[export] All {len(self._figures)} plots exported to {directory}")
