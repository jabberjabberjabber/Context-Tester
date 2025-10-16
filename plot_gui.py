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
        self.root.geometry("1800x1000")  # Wider to accommodate three-column layout

        # Data storage
        self.datasets = {}  # {display_name: dataframe}
        self.dataset_colors = {}
        self.dataset_markers = {}
        self.dataset_metadata = {}  # {display_name: metadata_dict}
        self.dataset_rounds = {}  # {display_name: list of individual round dicts}

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

        # Sidebar visibility
        self.left_sidebar_visible = True
        self.right_sidebar_visible = True

        # Layout options
        self.layout_mode = tk.StringVar(value='vertical')  # vertical, 1x1, 1x2, 2x1, 2x2, 3x1, 1x3, etc.

        # Plot style options
        self.show_smoothed = tk.BooleanVar(value=False)
        self.show_intervals = tk.BooleanVar(value=False)
        self.show_candlestick = tk.BooleanVar(value=False)
        self.show_trendline = tk.BooleanVar(value=False)

        # Default selected metrics
        self.default_metrics = ['vocabulary_diversity', 'cloze_score']

        self.setup_ui()

    def setup_ui(self):
        """Setup the main UI layout."""
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Create container for left sidebar, plot area, and right sidebar
        content_container = ttk.Frame(main_container)
        content_container.pack(fill=tk.BOTH, expand=True)

        # Left sidebar with toggle button
        self.left_sidebar = ttk.Frame(content_container)
        self.left_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 2), pady=5)

        # Left toggle button (collapses left sidebar)
        left_toggle_frame = ttk.Frame(content_container)
        left_toggle_frame.pack(side=tk.LEFT, fill=tk.Y, pady=5)
        self.left_toggle_btn = ttk.Button(
            left_toggle_frame,
            text="◀",
            width=2,
            command=self.toggle_left_sidebar
        )
        self.left_toggle_btn.pack(fill=tk.Y)

        # Center plot area (flexible - gets all extra space)
        plot_area = ttk.Frame(content_container)
        plot_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=5)

        # Right toggle button (collapses right sidebar)
        right_toggle_frame = ttk.Frame(content_container)
        right_toggle_frame.pack(side=tk.LEFT, fill=tk.Y, pady=5)
        self.right_toggle_btn = ttk.Button(
            right_toggle_frame,
            text="▶",
            width=2,
            command=self.toggle_right_sidebar
        )
        self.right_toggle_btn.pack(fill=tk.Y)

        # Right sidebar for layout controls
        self.right_sidebar = ttk.Frame(content_container)
        self.right_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 5), pady=5)

        # Setup left sidebar sections
        self.setup_load_section(self.left_sidebar)
        self.setup_dataset_section(self.left_sidebar)
        self.setup_metric_section(self.left_sidebar)

        # Setup right sidebar
        self.setup_layout_section(self.right_sidebar)
        self.setup_plot_options_section(self.right_sidebar)

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

    def setup_layout_section(self, parent):
        """Setup layout arrangement controls on right sidebar."""
        frame = ttk.LabelFrame(parent, text="Subplot Layout", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(frame, text="Arrangement:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        # Layout options with simple names
        layouts = [
            ('Vertical Stack', 'vertical'),
            ('1 × 2', '1x2'),
            ('2 × 2', '2x2'),
            ('1 Top / 2 Bottom', '1t_2b'),
            ('2 Top / 1 Bottom', '2t_1b'),
        ]

        for display_name, mode_value in layouts:
            rb = ttk.Radiobutton(
                frame,
                text=display_name,
                variable=self.layout_mode,
                value=mode_value,
                command=self.update_plot
            )
            rb.pack(anchor=tk.W, pady=2)

    def setup_plot_options_section(self, parent):
        """Setup plot style options on right sidebar."""
        frame = ttk.LabelFrame(parent, text="Plot Options", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(frame, text="Display:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        # Smoothed line option
        ttk.Checkbutton(
            frame,
            text="Smoothed Line",
            variable=self.show_smoothed,
            command=self.update_plot
        ).pack(anchor=tk.W, pady=2)

        # Individual rounds/intervals option
        ttk.Checkbutton(
            frame,
            text="Show Rounds (single dataset)",
            variable=self.show_intervals,
            command=self.update_plot
        ).pack(anchor=tk.W, pady=2)

        # Candlestick with std deviation
        ttk.Checkbutton(
            frame,
            text="Candlestick (± std dev)",
            variable=self.show_candlestick,
            command=self.update_plot
        ).pack(anchor=tk.W, pady=2)

        # Trend line
        ttk.Checkbutton(
            frame,
            text="Linear Trend Line",
            variable=self.show_trendline,
            command=self.update_plot
        ).pack(anchor=tk.W, pady=2)


    def setup_plot_area(self, parent):
        """Setup the matplotlib plot area."""
        self.fig = Figure(figsize=(10, 7), dpi=100)
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

    def _load_single_dataset(self, dataset_dir: Path) -> dict:
        """Load a single dataset directory.

        Args:
            dataset_dir: Path to dataset directory

        Returns:
            Dict with 'name', 'dataframe', 'metadata', 'rounds', or 'error' keys
        """
        from src.file_operations import load_experiment_metadata, load_individual_rounds_for_plotting

        try:
            # Load metadata
            metadata = load_experiment_metadata(dataset_dir)
            experiment_meta = metadata.get('experiment_metadata', {})

            model_name = experiment_meta.get('model_name', 'unknown')
            text_name = experiment_meta.get('source_text_name', 'unknown')
            model_id = experiment_meta.get('model_id', '')

            # Clean up model name
            if '/' in str(model_name):
                model_name = str(model_name).split('/')[-1]

            # Build display name
            parts = [str(model_name), str(text_name)]
            if model_id and str(model_id) != '':
                parts.append(str(model_id))
            name = ' - '.join(parts)

            # Load data
            individual_rounds, averaged_results, _ = load_individual_rounds_for_plotting(dataset_dir)

            # Create dataframe
            df = pd.DataFrame(averaged_results)
            df = df[df['context_length'].notna()].copy()

            if df.empty:
                return {
                    'name': name,
                    'error': 'No valid data in results'
                }

            return {
                'name': name,
                'dataframe': df,
                'metadata': metadata,
                'rounds': individual_rounds
            }

        except Exception as e:
            return {
                'name': dataset_dir.name,
                'error': str(e)
            }

    def load_results_folder(self):
        """Load results from either a single dataset directory or a parent folder containing multiple datasets."""
        folder = filedialog.askdirectory(title="Select results folder or dataset directory")
        if not folder:
            return

        folder_path = Path(folder)

        # Check if this is a single dataset directory (has metadata.json directly)
        if (folder_path / "metadata.json").exists():
            # Load single dataset
            loaded_datasets = [self._load_single_dataset(folder_path)]
        else:
            # Load all subdirectories as datasets
            loaded_datasets = load_all_results_from_folder(folder_path)

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
                rounds = dataset_info.get('rounds', [])

                # Handle duplicate names
                original_name = name
                counter = 1
                while name in self.datasets:
                    name = f"{original_name} ({counter})"
                    counter += 1

                # Store dataset
                self.datasets[name] = df
                self.dataset_metadata[name] = metadata
                self.dataset_rounds[name] = rounds

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
            self.dataset_rounds.clear()

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
        # Simple grid layouts
        if layout_mode == 'vertical':
            return {'use_gridspec': False, 'grid_rows': num_metrics, 'grid_cols': 1}
        elif layout_mode == '1x1':
            return {'use_gridspec': False, 'grid_rows': 1, 'grid_cols': 1}
        elif layout_mode == '1x2':
            return {'use_gridspec': False, 'grid_rows': 1, 'grid_cols': 2}
        elif layout_mode == '1x3':
            return {'use_gridspec': False, 'grid_rows': 1, 'grid_cols': 3}
        elif layout_mode == '2x1':
            return {'use_gridspec': False, 'grid_rows': 2, 'grid_cols': 1}
        elif layout_mode == '2x2':
            return {'use_gridspec': False, 'grid_rows': 2, 'grid_cols': 2}
        elif layout_mode == '3x1':
            return {'use_gridspec': False, 'grid_rows': 3, 'grid_cols': 1}

        # Custom arrangements using GridSpec
        elif layout_mode == '1t_1b':
            # 1 on top (full width), 1 on bottom (full width)
            positions = [
                {'row_start': 0, 'row_end': 1, 'col_start': 0, 'col_end': 2},
                {'row_start': 1, 'row_end': 2, 'col_start': 0, 'col_end': 2}
            ]
            return {'use_gridspec': True, 'grid_rows': 2, 'grid_cols': 2, 'positions': positions}

        elif layout_mode == '1t_2b':
            # 1 on top (full width), 2 on bottom (side by side)
            positions = [
                {'row_start': 0, 'row_end': 1, 'col_start': 0, 'col_end': 2},
                {'row_start': 1, 'row_end': 2, 'col_start': 0, 'col_end': 1},
                {'row_start': 1, 'row_end': 2, 'col_start': 1, 'col_end': 2}
            ]
            return {'use_gridspec': True, 'grid_rows': 2, 'grid_cols': 2, 'positions': positions}

        elif layout_mode == '2t_1b':
            # 2 on top (side by side), 1 on bottom (full width)
            positions = [
                {'row_start': 0, 'row_end': 1, 'col_start': 0, 'col_end': 1},
                {'row_start': 0, 'row_end': 1, 'col_start': 1, 'col_end': 2},
                {'row_start': 1, 'row_end': 2, 'col_start': 0, 'col_end': 2}
            ]
            return {'use_gridspec': True, 'grid_rows': 2, 'grid_cols': 2, 'positions': positions}

        else:
            # Default to vertical
            return {'use_gridspec': False, 'grid_rows': num_metrics, 'grid_cols': 1}

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

            # Set default plot style
            linestyle = '-'
            marker = True
            line_width = 2.0
            marker_size = 6.0

            # Set figure size (will scale with window)
            self.fig.set_size_inches(10, 7)

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

                    # Plot individual rounds if enabled (only works with single dataset)
                    if self.show_intervals.get() and len(selected_datasets) == 1:
                        self._plot_individual_rounds(ax, dataset_name, metric, self.dataset_colors[dataset_name])

                    # Plot candlestick (error bars with std deviation)
                    if self.show_candlestick.get():
                        self._plot_candlestick(ax, df, x_data, y_data, metric, self.dataset_colors[dataset_name])

                    # Plot main line
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

                    # Plot smoothed line if enabled
                    if self.show_smoothed.get():
                        self._plot_smoothed_line(ax, x_data, y_data, self.dataset_colors[dataset_name], dataset_name)

                    # Plot trend line if enabled
                    if self.show_trendline.get():
                        self._plot_trendline(ax, x_data, y_data, self.dataset_colors[dataset_name])

                # Add ground truth reference line if available
                # Add ground truth for each selected dataset using its matching color
                for dataset_name in selected_datasets:
                    if dataset_name in self.dataset_metadata:
                        metadata = self.dataset_metadata[dataset_name]
                        ground_truth = metadata.get('ground_truth_analysis', {})
                        if metric in ground_truth:
                            gt_value = ground_truth[metric]
                            # Use the dataset's color for its ground truth line
                            ax.axhline(y=gt_value, color=self.dataset_colors[dataset_name],
                                     linestyle='--', linewidth=2,
                                     label=f'{dataset_name} GT', alpha=0.7)

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

    def toggle_left_sidebar(self):
        """Toggle left sidebar visibility."""
        if self.left_sidebar_visible:
            # Hide sidebar
            self.left_sidebar.pack_forget()
            self.left_toggle_btn.config(text="▶")
            self.left_sidebar_visible = False
        else:
            # Show sidebar
            self.left_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 2), pady=5, before=self.left_toggle_btn.master)
            self.left_toggle_btn.config(text="◀")
            self.left_sidebar_visible = True

    def toggle_right_sidebar(self):
        """Toggle right sidebar visibility."""
        if self.right_sidebar_visible:
            # Hide sidebar
            self.right_sidebar.pack_forget()
            self.right_toggle_btn.config(text="◀")
            self.right_sidebar_visible = False
        else:
            # Show sidebar
            self.right_sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 5), pady=5)
            self.right_toggle_btn.config(text="▶")
            self.right_sidebar_visible = True

    def get_colors_and_markers(self):
        """Get color and marker lists for datasets."""
        colors = [
            '#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea', '#c2410c',
            '#0891b2', '#e11d48', '#65a30d', '#d97706', '#7c3aed', '#ea580c'
        ]
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'H']
        return colors, markers

    def _plot_individual_rounds(self, ax, dataset_name, metric, color):
        """Plot individual rounds as semi-transparent lines."""
        if dataset_name not in self.dataset_rounds:
            return

        rounds_data = self.dataset_rounds[dataset_name]
        if not rounds_data:
            return

        # Group rounds by context length
        from collections import defaultdict
        context_rounds = defaultdict(list)

        for round_data in rounds_data:
            if metric in round_data and round_data[metric] is not None:
                ctx = round_data.get('context_length')
                if ctx is not None:
                    context_rounds[ctx].append(round_data[metric])

        # Plot each individual round
        contexts = sorted(context_rounds.keys())
        num_rounds = max(len(vals) for vals in context_rounds.values()) if context_rounds else 0

        for round_idx in range(num_rounds):
            round_x = []
            round_y = []
            for ctx in contexts:
                if round_idx < len(context_rounds[ctx]):
                    round_x.append(ctx)
                    round_y.append(context_rounds[ctx][round_idx])

            if round_x and round_y:
                ax.plot(round_x, round_y, color=color, alpha=0.2, linewidth=1, linestyle='-')

    def _plot_candlestick(self, ax, df, x_data, y_data, metric, color):
        """Plot candlestick with std deviation error bars."""
        # Need std deviation column
        std_col = f'{metric}_std'
        if std_col not in df.columns:
            # Try computing from rounds if available
            return

        mask = df[metric].notna() & df['context_length'].notna() & df[std_col].notna()
        if not mask.any():
            return

        x = df.loc[mask, 'context_length']
        y = df.loc[mask, metric]
        std = df.loc[mask, std_col]

        # Plot error bars
        ax.errorbar(x, y, yerr=std, color=color, alpha=0.5,
                   fmt='none', capsize=5, capthick=2, elinewidth=2)

    def _plot_smoothed_line(self, ax, x_data, y_data, color, label):
        """Plot smoothed line using moving average or spline."""
        import numpy as np
        from scipy.interpolate import make_interp_spline

        if len(x_data) < 3:
            return  # Need at least 3 points for spline

        # Convert to numpy arrays and sort
        x_arr = np.array(x_data)
        y_arr = np.array(y_data)
        sort_idx = np.argsort(x_arr)
        x_sorted = x_arr[sort_idx]
        y_sorted = y_arr[sort_idx]

        # Create spline
        try:
            # Use log scale for x
            x_log = np.log10(x_sorted)
            spline = make_interp_spline(x_log, y_sorted, k=min(3, len(x_sorted)-1))

            # Generate smooth curve
            x_log_smooth = np.linspace(x_log.min(), x_log.max(), 300)
            x_smooth = 10 ** x_log_smooth
            y_smooth = spline(x_log_smooth)

            ax.plot(x_smooth, y_smooth, color=color, alpha=0.4,
                   linewidth=3, linestyle='-', label=f'{label} (smoothed)')
        except Exception:
            # If spline fails, skip smoothing
            pass

    def _plot_trendline(self, ax, x_data, y_data, color):
        """Plot linear trend line in log-linear space."""
        import numpy as np

        if len(x_data) < 2:
            return

        # Use log of x for trend line
        x_log = np.log10(np.array(x_data))
        y_arr = np.array(y_data)

        # Linear regression
        coeffs = np.polyfit(x_log, y_arr, 1)
        trend_y = np.polyval(coeffs, x_log)

        # Sort for plotting
        sort_idx = np.argsort(x_data)
        x_sorted = np.array(x_data)[sort_idx]
        trend_y_sorted = trend_y[sort_idx]

        ax.plot(x_sorted, trend_y_sorted, color=color,
               linestyle=':', linewidth=2, alpha=0.7, label='_nolegend_')


def main():
    """Main entry point."""
    root = tk.Tk()
    app = PlotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
