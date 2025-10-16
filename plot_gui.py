#!/usr/bin/env python3
"""
Interactive GUI for Context Tester Results Visualization

Clean, simple design focused on easy data exploration.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import traceback

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import pandas as pd

from src.file_operations import load_all_results_from_folder
from src.metadata_utils import format_metadata_for_tooltip, format_metadata_for_popup


class PlotGUI:
    """Interactive plotting GUI for Context Tester results."""

    def __init__(self, root):
        self.root = root
        self.root.title("Context Tester - Interactive Plots")
        self.root.geometry("1600x1000")

        # Data storage
        self.datasets = {}  # {display_name: dataframe}
        self.dataset_colors = {}
        self.dataset_markers = {}
        self.dataset_metadata = {}  # {display_name: metadata_dict}

        # Available metrics
        self.available_metrics = [
            'vocabulary_diversity',
            'cloze_score',
            'adjacent_coherence',
            'bigram_repetition_rate',
            'avg_sentence_length',
            'sentence_length_variance',
            'pct_unfamiliar_words',
            'trigram_repetition_rate',
            'unique_word_ratio_100',
            'word_entropy',
            'char_entropy',
            'comma_density',
            'semicolon_density',
            'question_density',
            'exclamation_density',
            'avg_syllables_per_word',
            'long_word_ratio',
            'function_word_ratio',
            'sentence_length_skewness',
            'sentence_length_kurtosis',
            'avg_word_length',
            'global_coherence',
            'local_coherence_3sent',
            'coherence_variance'
        ]

        self.metric_labels = {
            'vocabulary_diversity': 'Vocabulary Diversity',
            'cloze_score': 'Cloze Score',
            'adjacent_coherence': 'Adjacent Similarity',
            'bigram_repetition_rate': 'Bigram Repetition',
            'avg_sentence_length': 'Avg Sentence Length',
            'sentence_length_variance': 'Sentence Length Variance',
            'pct_unfamiliar_words': 'Unfamiliar Words %',
            'trigram_repetition_rate': 'Trigram Repetition',
            'unique_word_ratio_100': 'Unique Word Ratio',
            'word_entropy': 'Word Entropy',
            'char_entropy': 'Character Entropy',
            'comma_density': 'Comma Density',
            'semicolon_density': 'Semicolon Density',
            'question_density': 'Question Density',
            'exclamation_density': 'Exclamation Density',
            'avg_syllables_per_word': 'Avg Syllables/Word',
            'long_word_ratio': 'Long Word Ratio',
            'function_word_ratio': 'Function Word Ratio',
            'sentence_length_skewness': 'Sentence Length Skewness',
            'sentence_length_kurtosis': 'Sentence Length Kurtosis',
            'avg_word_length': 'Avg Word Length',
            'global_coherence': 'Global Coherence',
            'local_coherence_3sent': 'Local Coherence (3-sent)',
            'coherence_variance': 'Coherence Variance'
        }

        # UI state
        self.dataset_vars = {}  # {name: BooleanVar}
        self.metric_vars = {}   # {metric: BooleanVar}
        self.invert_vars = {}   # {metric: BooleanVar}

        # Plot style options
        self.plot_style = tk.StringVar(value='lines_markers')  # lines_markers, lines_only, markers_only
        self.line_width = tk.DoubleVar(value=2.0)
        self.marker_size = tk.DoubleVar(value=6.0)

        # Layout options
        self.layout_mode = tk.StringVar(value='vertical')  # vertical, grid_2x2, grid_3x1, grid_1x3, custom, etc.
        self.custom_rows = tk.IntVar(value=2)
        self.custom_cols = tk.IntVar(value=2)
        self.fig_width = tk.DoubleVar(value=12.0)
        self.fig_height = tk.DoubleVar(value=8.0)

        # Subplot size customization
        self.subplot_spans = {}  # {metric_index: (row_span, col_span)}

        # Default selected metrics
        self.default_metrics = ['vocabulary_diversity', 'cloze_score']

        self.setup_ui()

    def setup_toolbar(self, parent):
        """Setup the top toolbar with plot style and layout controls."""
        toolbar = ttk.Frame(parent, relief=tk.RAISED, borderwidth=1)
        toolbar.pack(fill=tk.X, padx=5, pady=5)

        # Left section - Plot Style
        style_frame = ttk.LabelFrame(toolbar, text="Plot Style", padding=5)
        style_frame.pack(side=tk.LEFT, padx=5)

        # Plot type
        ttk.Label(style_frame, text="Type:").grid(row=0, column=0, padx=2, sticky=tk.W)
        plot_type_combo = ttk.Combobox(
            style_frame,
            textvariable=self.plot_style,
            values=['lines_markers', 'lines_only', 'markers_only'],
            state='readonly',
            width=13
        )
        plot_type_combo.grid(row=0, column=1, padx=2)
        plot_type_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plot())

        # Line width
        ttk.Label(style_frame, text="Line:").grid(row=0, column=2, padx=2, sticky=tk.W)
        ttk.Scale(
            style_frame,
            from_=0.5,
            to=5.0,
            variable=self.line_width,
            orient=tk.HORIZONTAL,
            length=80,
            command=lambda _: self.update_plot()
        ).grid(row=0, column=3, padx=2)
        self.line_width_label = ttk.Label(style_frame, text=f"{self.line_width.get():.1f}", width=3)
        self.line_width_label.grid(row=0, column=4, padx=2)
        self.line_width.trace_add('write', lambda *_: self.line_width_label.config(text=f"{self.line_width.get():.1f}"))

        # Marker size
        ttk.Label(style_frame, text="Marker:").grid(row=0, column=5, padx=2, sticky=tk.W)
        ttk.Scale(
            style_frame,
            from_=2.0,
            to=15.0,
            variable=self.marker_size,
            orient=tk.HORIZONTAL,
            length=80,
            command=lambda _: self.update_plot()
        ).grid(row=0, column=6, padx=2)
        self.marker_size_label = ttk.Label(style_frame, text=f"{self.marker_size.get():.1f}", width=3)
        self.marker_size_label.grid(row=0, column=7, padx=2)
        self.marker_size.trace_add('write', lambda *_: self.marker_size_label.config(text=f"{self.marker_size.get():.1f}"))

        # Middle section - Layout
        layout_frame = ttk.LabelFrame(toolbar, text="Layout", padding=5)
        layout_frame.pack(side=tk.LEFT, padx=5)

        # Layout mode
        ttk.Label(layout_frame, text="Grid:").grid(row=0, column=0, padx=2, sticky=tk.W)
        layout_combo = ttk.Combobox(
            layout_frame,
            textvariable=self.layout_mode,
            values=['vertical', 'grid_2x2', 'grid_3x1', 'grid_1x3', 'grid_2x1', 'grid_1x2',
                    'top_large', 'bottom_large', 'left_large', 'right_large', 'custom'],
            state='readonly',
            width=12
        )
        layout_combo.grid(row=0, column=1, padx=2)
        layout_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plot())

        # Custom rows/cols (show always but disable when not custom)
        ttk.Label(layout_frame, text="R:").grid(row=0, column=2, padx=2, sticky=tk.W)
        self.rows_spin = ttk.Spinbox(
            layout_frame,
            from_=1,
            to=10,
            textvariable=self.custom_rows,
            width=4,
            command=self.update_plot
        )
        self.rows_spin.grid(row=0, column=3, padx=2)

        ttk.Label(layout_frame, text="C:").grid(row=0, column=4, padx=2, sticky=tk.W)
        self.cols_spin = ttk.Spinbox(
            layout_frame,
            from_=1,
            to=10,
            textvariable=self.custom_cols,
            width=4,
            command=self.update_plot
        )
        self.cols_spin.grid(row=0, column=5, padx=2)

        # Disable rows/cols if not custom
        def update_custom_state(*args):
            state = 'normal' if self.layout_mode.get() == 'custom' else 'disabled'
            self.rows_spin.config(state=state)
            self.cols_spin.config(state=state)

        self.layout_mode.trace_add('write', update_custom_state)
        update_custom_state()

        # Right section - Figure Size
        size_frame = ttk.LabelFrame(toolbar, text="Figure Size", padding=5)
        size_frame.pack(side=tk.LEFT, padx=5)

        # Width
        ttk.Label(size_frame, text="W:").grid(row=0, column=0, padx=2, sticky=tk.W)
        ttk.Scale(
            size_frame,
            from_=6.0,
            to=20.0,
            variable=self.fig_width,
            orient=tk.HORIZONTAL,
            length=80,
            command=lambda _: self.update_plot()
        ).grid(row=0, column=1, padx=2)
        self.fig_width_label = ttk.Label(size_frame, text=f"{self.fig_width.get():.1f}", width=3)
        self.fig_width_label.grid(row=0, column=2, padx=2)
        self.fig_width.trace_add('write', lambda *_: self.fig_width_label.config(text=f"{self.fig_width.get():.1f}"))

        # Height
        ttk.Label(size_frame, text="H:").grid(row=0, column=3, padx=2, sticky=tk.W)
        ttk.Scale(
            size_frame,
            from_=4.0,
            to=20.0,
            variable=self.fig_height,
            orient=tk.HORIZONTAL,
            length=80,
            command=lambda _: self.update_plot()
        ).grid(row=0, column=4, padx=2)
        self.fig_height_label = ttk.Label(size_frame, text=f"{self.fig_height.get():.1f}", width=3)
        self.fig_height_label.grid(row=0, column=5, padx=2)
        self.fig_height.trace_add('write', lambda *_: self.fig_height_label.config(text=f"{self.fig_height.get():.1f}"))

    def setup_ui(self):
        """Setup the main UI layout."""
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Top toolbar
        #self.setup_toolbar(main_container)

        # Create container for sidebar and plot area
        content_container = ttk.Frame(main_container)
        content_container.pack(fill=tk.BOTH, expand=True)

        # Left sidebar (fixed width)
        sidebar = ttk.Frame(content_container, width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        #sidebar.pack_propagate(False)  # Maintain fixed width

        # Right plot area (flexible)
        plot_area = ttk.Frame(content_container)
        plot_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Setup sidebar sections
        self.setup_load_section(sidebar)
        self.setup_dataset_section(sidebar)
        self.setup_metric_section(sidebar)
        self.setup_export_section(sidebar)

        # Setup plot area
        self.setup_plot_area(plot_area)

    def setup_load_section(self, parent):
        """Setup data loading section."""
        frame = ttk.LabelFrame(parent, text="Load Data", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            frame,
            text="Load Results Folder",
            command=self.load_results_folder
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            frame,
            text="Clear All",
            command=self.clear_all
        ).pack(fill=tk.X, pady=2)

        self.status_label = ttk.Label(frame, text="No data loaded", foreground="gray")
        self.status_label.pack(fill=tk.X, pady=(5, 0))

    def setup_dataset_section(self, parent):
        """Setup dataset selection section."""
        frame = ttk.LabelFrame(parent, text="Datasets", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        # Scrollable list with fixed height
        scroll_frame = ttk.Frame(frame)
        scroll_frame.pack(fill=tk.X)

        canvas = tk.Canvas(scroll_frame, height=150)
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        self.dataset_container = ttk.Frame(canvas)

        self.dataset_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=self.dataset_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Make canvas window fill width
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def setup_metric_section(self, parent):
        """Setup metric selection section."""
        frame = ttk.LabelFrame(parent, text="Metrics", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        # Non-scrollable list - just pack all metrics
        self.metric_container = ttk.Frame(frame)
        self.metric_container.pack(fill=tk.X)

        # Create metric checkboxes
        for metric in self.available_metrics:
            self.add_metric_row(metric)

    def setup_export_section(self, parent):
        """Setup export controls."""
        frame = ttk.LabelFrame(parent, text="Export", padding=10)
        frame.pack(fill=tk.X)

        ttk.Button(
            frame,
            text="Export to PNG",
            command=self.export_plot
        ).pack(fill=tk.X)

    def setup_plot_area(self, parent):
        """Setup the matplotlib plot area."""
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)

        # Toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        # Canvas
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Setup hover annotation
        self.hover_annotation = None
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)

        # Initial message
        self.show_message("Load data to begin")

    def add_metric_row(self, metric):
        """Add a metric checkbox row."""
        row = ttk.Frame(self.metric_container)
        row.pack(fill=tk.X, pady=1)

        # Metric checkbox
        var = tk.BooleanVar(value=(metric in self.default_metrics))
        self.metric_vars[metric] = var

        cb = ttk.Checkbutton(
            row,
            text=self.metric_labels[metric],
            variable=var,
            command=self.update_plot
        )
        cb.pack(side=tk.LEFT)

        # Invert Y checkbox
        invert_var = tk.BooleanVar(value=(metric == 'vocabulary_diversity'))
        self.invert_vars[metric] = invert_var

        invert_cb = ttk.Checkbutton(
            row,
            text="⇅",  # Up-down arrow
            variable=invert_var,
            command=self.update_plot,
            width=3
        )
        invert_cb.pack(side=tk.RIGHT)

    def add_dataset_row(self, name):
        """Add a dataset checkbox row."""
        row = ttk.Frame(self.dataset_container)
        row.pack(fill=tk.X, pady=1)

        # Color indicator
        color_box = tk.Canvas(row, width=16, height=16, highlightthickness=0)
        color_box.pack(side=tk.LEFT, padx=(0, 5))
        color_box.create_rectangle(
            2, 2, 14, 14,
            fill=self.dataset_colors[name],
            outline=self.dataset_colors[name]
        )

        # Dataset checkbox
        var = tk.BooleanVar(value=True)
        self.dataset_vars[name] = var

        cb = ttk.Checkbutton(
            row,
            text=name,
            variable=var,
            command=self.update_plot
        )
        cb.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def load_results_folder(self):
        """Load all result directories from a parent folder."""
        folder = filedialog.askdirectory(title="Select results folder")
        if not folder:
            return

        # Use file_operations to load datasets
        loaded_datasets = load_all_results_from_folder(Path(folder))

        if not loaded_datasets:
            messagebox.showwarning(
                "No Data",
                "No valid result directories found.\n\n"
                "Looking for directories with metadata.json files."
            )
            return

        loaded_count = 0
        error_count = 0

        for dataset_info in loaded_datasets:
            if 'error' in dataset_info:
                error_count += 1
                print(f"Error loading {dataset_info['name']}: {dataset_info['error']}")
                continue

            try:
                name = dataset_info['name']
                df = dataset_info['dataframe']
                metadata = dataset_info['metadata']

                # Handle duplicate names
                original_name = name
                counter = 1
                while name in self.datasets:
                    name = f"{original_name} ({counter})"
                    counter += 1

                # Store dataset
                self.datasets[name] = df
                self.dataset_metadata[name] = metadata

                # Assign color and marker
                colors, markers = self.get_colors_and_markers()
                idx = len(self.dataset_colors)
                self.dataset_colors[name] = colors[idx % len(colors)]
                self.dataset_markers[name] = markers[idx % len(markers)]

                # Add UI element
                self.add_dataset_row(name)
                loaded_count += 1

            except Exception as e:
                error_count += 1
                print(f"Error processing {dataset_info.get('name', 'unknown')}: {e}")
                traceback.print_exc()

        self.update_status()
        self.update_plot()

        if error_count > 0:
            messagebox.showinfo(
                "Load Complete",
                f"Loaded {loaded_count} dataset(s).\n"
                f"Skipped {error_count} invalid/broken dataset(s)."
            )

    def clear_all(self):
        """Clear all loaded data."""
        if not self.datasets:
            return

        if messagebox.askyesno("Confirm", "Clear all loaded data?"):
            self.datasets.clear()
            self.dataset_vars.clear()
            self.dataset_colors.clear()
            self.dataset_markers.clear()
            self.dataset_metadata.clear()

            # Clear UI
            for widget in self.dataset_container.winfo_children():
                widget.destroy()

            self.update_status()
            self.update_plot()

    def update_status(self):
        """Update status label."""
        count = len(self.datasets)
        if count == 0:
            self.status_label.config(text="No data loaded", foreground="gray")
        else:
            self.status_label.config(
                text=f"{count} dataset{'s' if count != 1 else ''} loaded",
                foreground="green"
            )

    def show_message(self, text):
        """Show a message in the plot area."""
        self.fig.clear()
        self.fig.text(
            0.5, 0.5, text,
            ha='center', va='center',
            fontsize=14, color='gray'
        )
        self.canvas.draw()

    def _get_subplot_specs(self, layout_mode, num_metrics):
        """
        Get subplot specifications for different layout modes.

        Returns dict with:
        - use_gridspec: bool - whether to use GridSpec (True) or simple grid (False)
        - grid_rows: int - number of rows in the grid
        - grid_cols: int - number of columns in the grid
        - positions: list of dicts with row_start, row_end, col_start, col_end (only if use_gridspec=True)
        """
        import math

        if layout_mode == 'vertical':
            return {
                'use_gridspec': False,
                'grid_rows': num_metrics,
                'grid_cols': 1,
            }
        elif layout_mode == 'grid_2x2':
            return {
                'use_gridspec': False,
                'grid_rows': 2,
                'grid_cols': 2,
            }
        elif layout_mode == 'grid_3x1':
            return {
                'use_gridspec': False,
                'grid_rows': 3,
                'grid_cols': 1,
            }
        elif layout_mode == 'grid_1x3':
            return {
                'use_gridspec': False,
                'grid_rows': 1,
                'grid_cols': 3,
            }
        elif layout_mode == 'grid_2x1':
            return {
                'use_gridspec': False,
                'grid_rows': 2,
                'grid_cols': 1,
            }
        elif layout_mode == 'grid_1x2':
            return {
                'use_gridspec': False,
                'grid_rows': 1,
                'grid_cols': 2,
            }
        elif layout_mode == 'top_large':
            # 1 large plot on top (spans 2 rows), remaining plots in single rows below
            positions = [{'row_start': 0, 'row_end': 2, 'col_start': 0, 'col_end': 2}]
            for i in range(1, num_metrics):
                row = 2 + i - 1
                positions.append({'row_start': row, 'row_end': row + 1, 'col_start': 0, 'col_end': 2})
            return {
                'use_gridspec': True,
                'grid_rows': 2 + max(0, num_metrics - 1),
                'grid_cols': 2,
                'positions': positions,
            }
        elif layout_mode == 'bottom_large':
            # Smaller plots on top, 1 large plot at bottom (spans 2 rows)
            positions = []
            for i in range(num_metrics - 1):
                positions.append({'row_start': i, 'row_end': i + 1, 'col_start': 0, 'col_end': 2})
            # Last plot is large
            row = max(0, num_metrics - 1)
            positions.append({'row_start': row, 'row_end': row + 2, 'col_start': 0, 'col_end': 2})
            return {
                'use_gridspec': True,
                'grid_rows': max(num_metrics - 1, 0) + 2,
                'grid_cols': 2,
                'positions': positions,
            }
        elif layout_mode == 'left_large':
            # 1 large plot on left (spans 2 columns), remaining plots stacked on right
            positions = [{'row_start': 0, 'row_end': num_metrics, 'col_start': 0, 'col_end': 2}]
            for i in range(1, num_metrics):
                positions.append({'row_start': i - 1, 'row_end': i, 'col_start': 2, 'col_end': 3})
            return {
                'use_gridspec': True,
                'grid_rows': max(num_metrics, 1),
                'grid_cols': 3,
                'positions': positions,
            }
        elif layout_mode == 'right_large':
            # Smaller plots stacked on left, 1 large plot on right (spans all rows)
            positions = []
            for i in range(num_metrics - 1):
                positions.append({'row_start': i, 'row_end': i + 1, 'col_start': 0, 'col_end': 1})
            # Last plot is large on the right
            positions.append({'row_start': 0, 'row_end': max(num_metrics - 1, 1), 'col_start': 1, 'col_end': 3})
            return {
                'use_gridspec': True,
                'grid_rows': max(num_metrics - 1, 1),
                'grid_cols': 3,
                'positions': positions,
            }
        elif layout_mode == 'custom':
            rows = self.custom_rows.get()
            cols = self.custom_cols.get()
            return {
                'use_gridspec': False,
                'grid_rows': rows,
                'grid_cols': cols,
            }
        else:
            # Default to vertical
            return {
                'use_gridspec': False,
                'grid_rows': num_metrics,
                'grid_cols': 1,
            }

    def update_plot(self):
        """Update the plots based on current selections."""
        try:
            # Get selected datasets and metrics
            selected_datasets = [
                name for name, var in self.dataset_vars.items()
                if var.get()
            ]
            selected_metrics = [
                metric for metric, var in self.metric_vars.items()
                if var.get()
            ]

            # Clear figure
            self.fig.clear()

            # Check for data
            if not selected_datasets:
                self.show_message("No datasets selected")
                return

            if not selected_metrics:
                self.show_message("No metrics selected")
                return

            # Get plot style settings
            plot_style = self.plot_style.get()
            if plot_style == 'lines_markers':
                linestyle = '-'
                marker = True
            elif plot_style == 'lines_only':
                linestyle = '-'
                marker = False
            else:  # markers_only
                linestyle = 'None'
                marker = True

            line_width = self.line_width.get()
            marker_size = self.marker_size.get()

            # Set figure size
            self.fig.set_size_inches(self.fig_width.get(), self.fig_height.get())

            # Store line objects for hover detection
            self.plot_lines = {}  # {line_object: dataset_name}

            # Determine subplot layout
            num_metrics = len(selected_metrics)
            layout_mode = self.layout_mode.get()

            # Get subplot specifications based on layout mode
            subplot_specs = self._get_subplot_specs(layout_mode, num_metrics)

            # Create GridSpec for flexible layouts
            if subplot_specs['use_gridspec']:
                gs = GridSpec(
                    subplot_specs['grid_rows'],
                    subplot_specs['grid_cols'],
                    figure=self.fig
                )

                # Create subplots with specified spans
                axes = []
                for idx, metric in enumerate(selected_metrics):
                    if idx >= len(subplot_specs['positions']):
                        break  # Skip if we have more metrics than positions

                    pos = subplot_specs['positions'][idx]
                    ax = self.fig.add_subplot(
                        gs[pos['row_start']:pos['row_end'],
                           pos['col_start']:pos['col_end']]
                    )
                    axes.append((ax, metric))
            else:
                # Use simple grid layout
                rows, cols = subplot_specs['grid_rows'], subplot_specs['grid_cols']
                axes = []
                for idx, metric in enumerate(selected_metrics):
                    if idx >= rows * cols:
                        break
                    ax = self.fig.add_subplot(rows, cols, idx + 1)
                    axes.append((ax, metric))

            # Plot each metric
            for ax, metric in axes:

                # Plot each selected dataset
                for dataset_name in selected_datasets:
                    if dataset_name not in self.datasets:
                        continue

                    df = self.datasets[dataset_name]

                    if metric not in df.columns:
                        continue

                    # Filter valid data
                    mask = df[metric].notna() & df['context_length'].notna()
                    if not mask.any():
                        continue

                    x_data = df.loc[mask, 'context_length']
                    y_data = df.loc[mask, metric]

                    # Determine marker based on style
                    marker_symbol = self.dataset_markers[dataset_name] if marker else None

                    # Plot and store line object
                    line, = ax.plot(
                        x_data, y_data,
                        marker=marker_symbol,
                        color=self.dataset_colors[dataset_name],
                        linewidth=line_width,
                        markersize=marker_size,
                        label=dataset_name,
                        linestyle=linestyle,
                        picker=5  # Enable picking with 5pt tolerance
                    )
                    self.plot_lines[line] = dataset_name

                # Add ground truth reference line if available
                # Check if any selected dataset has ground truth for this metric
                for dataset_name in selected_datasets:
                    if dataset_name in self.dataset_metadata:
                        metadata = self.dataset_metadata[dataset_name]
                        ground_truth = metadata.get('ground_truth_analysis', {})
                        if metric in ground_truth:
                            gt_value = ground_truth[metric]
                            ax.axhline(y=gt_value, color='green', linestyle='--',
                                     linewidth=2, label='Ground Truth', alpha=0.7)
                            break  # Only add once per plot

                # Format subplot
                ax.set_xscale('log')
                ax.set_xlabel('Context Length', fontsize=10)
                ax.set_ylabel(self.metric_labels[metric], fontsize=10)
                ax.set_title(self.metric_labels[metric], fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Invert Y-axis if requested
                if self.invert_vars[metric].get():
                    ax.invert_yaxis()

                # Format x-axis
                all_contexts = set()
                for dataset_name in selected_datasets:
                    if dataset_name in self.datasets:
                        df = self.datasets[dataset_name]
                        all_contexts.update(df['context_length'].dropna().unique())

                if all_contexts:
                    unique_contexts = sorted(all_contexts)
                    ax.set_xticks(unique_contexts)
                    ax.set_xticklabels(
                        [f'{int(x/1024)}K' if x >= 1024 else str(int(x))
                         for x in unique_contexts],
                        fontsize=8
                    )

                # Legend (only if multiple datasets)
                if len(selected_datasets) > 1:
                    ax.legend(loc='best', fontsize=8)

            self.fig.tight_layout()
            self.canvas.draw_idle()

            # Force canvas resize to make figure fit properly
            self._trigger_canvas_resize()

        except Exception as e:
            messagebox.showerror("Plot Error", f"Error updating plot:\n{str(e)}")
            traceback.print_exc()

    def _trigger_canvas_resize(self):
        """Force a resize event on the canvas to make figure fit properly."""
        canvas_widget = self.canvas.get_tk_widget()
        # Generate a Configure event with current dimensions
        canvas_widget.event_generate(
            "<Configure>",
            width=canvas_widget.winfo_width(),
            height=canvas_widget.winfo_height()
        )

    def on_hover(self, event):
        """Handle mouse hover over plot lines."""
        if event.inaxes is None:
            # Clear any existing annotation
            if self.hover_annotation is not None and self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.canvas.draw_idle()
            return

        # Check if hovering over any line
        for line, dataset_name in self.plot_lines.items():
            if line.axes != event.inaxes:
                continue

            contains, _ = line.contains(event)
            if contains:
                # Show metadata tooltip
                self.show_hover_tooltip(event, dataset_name)
                return

        # Clear annotation if not hovering over any line
        if self.hover_annotation is not None and self.hover_annotation.get_visible():
            self.hover_annotation.set_visible(False)
            self.canvas.draw_idle()

    def show_hover_tooltip(self, event, dataset_name):
        """Show tooltip with metadata when hovering over a line."""
        if dataset_name not in self.dataset_metadata:
            return

        metadata = self.dataset_metadata[dataset_name]

        # Build tooltip text using utility function
        tooltip_text = f"{dataset_name}\n" + format_metadata_for_tooltip(metadata)

        # Create or update annotation
        if self.hover_annotation is None or self.hover_annotation.axes != event.inaxes:
            # Create new annotation (first time or switching axes)
            if self.hover_annotation is not None:
                self.hover_annotation.set_visible(False)

            self.hover_annotation = event.inaxes.annotate(
                tooltip_text,
                xy=(event.xdata, event.ydata),
                xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.9, edgecolor="black"),
                fontsize=8,
                family='monospace',
                zorder=1000
            )
        else:
            # Update existing annotation
            self.hover_annotation.set_text(tooltip_text)
            self.hover_annotation.xy = (event.xdata, event.ydata)
            self.hover_annotation.set_visible(True)

        self.canvas.draw_idle()

    def show_metadata(self, dataset_name):
        """Show metadata for a dataset in a popup window."""
        if dataset_name not in self.dataset_metadata:
            messagebox.showinfo("No Metadata", f"No metadata available for {dataset_name}")
            return

        metadata = self.dataset_metadata[dataset_name]

        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title(f"Metadata: {dataset_name}")
        popup.geometry("600x500")

        # Add scrollable text widget
        frame = ttk.Frame(popup, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Create text widget with scrollbar
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            font=("Courier", 10),
            padx=10,
            pady=10
        )
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        # Format and display metadata using utility function
        formatted_text = format_metadata_for_popup(metadata)
        text_widget.insert(tk.END, formatted_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only

        # Close button
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            btn_frame,
            text="Close",
            command=popup.destroy
        ).pack(side=tk.RIGHT)

    def export_plot(self):
        """Export current plot to PNG."""
        if not self.datasets:
            messagebox.showwarning("No Data", "No data to export")
            return

        filename = filedialog.asksaveasfilename(
            title="Export plot",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ]
        )

        if not filename:
            return

        try:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{str(e)}")

    def get_colors_and_markers(self):
        """Get color and marker lists for datasets."""
        colors = [
            '#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea', '#c2410c',
            '#0891b2', '#e11d48', '#65a30d', '#d97706', '#7c3aed', '#ea580c'
        ]
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'H']
        return colors, markers


def main():
    """Main entry point."""
    root = tk.Tk()
    app = PlotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
