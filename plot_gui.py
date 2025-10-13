#!/usr/bin/env python3
"""
Interactive GUI for Context Tester Results Visualization

Features:
- Load multiple CSV files or result directories
- Toggle datasets and metrics on/off in real-time
- Interactive matplotlib plots with zoom/pan
- Customize plot appearance and axis ranges
- Export plots to PNG
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import traceback

from src.file_operations import (
    load_csv_for_plotting,
    load_individual_rounds_for_plotting,
    load_experiment_metadata,
    get_dataset_name_from_csv
)


class PlotGUI:
    """Interactive plotting GUI for Context Tester results."""

    def __init__(self, root):
        self.root = root
        self.root.title("Context Tester - Interactive Plots")
        self.root.geometry("1400x900")

        # Data storage
        self.datasets = {}  # {name: dataframe}
        self.dataset_colors = {}
        self.dataset_markers = {}

        # Plot configuration
        self.available_metrics = [
            'vocabulary_diversity',
            'cloze_score',
            'adjacent_coherence',
            'bigram_repetition_rate',
            'avg_sentence_length',
            'sentence_length_variance',
            'pct_unfamiliar_words'
        ]

        self.metric_labels = {
            'vocabulary_diversity': 'Vocabulary Diversity',
            'cloze_score': 'Cloze Score',
            'adjacent_coherence': 'Adjacent Similarity',
            'bigram_repetition_rate': 'Bigram Repetition Rate',
            'avg_sentence_length': 'Avg Sentence Length',
            'sentence_length_variance': 'Sentence Length Variance',
            'pct_unfamiliar_words': 'Unfamiliar Words %'
        }

        self.metric_descriptions = {
            'vocabulary_diversity': 'Higher = Less Diverse',
            'cloze_score': 'Higher = More Basic',
            'adjacent_coherence': 'Higher = More Similar',
            'bigram_repetition_rate': 'Higher = More Repetitive',
            'avg_sentence_length': 'Average words per sentence',
            'sentence_length_variance': 'Sentence length consistency',
            'pct_unfamiliar_words': 'Percentage of difficult words'
        }

        # UI state
        self.dataset_vars = {}  # {name: BooleanVar}
        self.metric_vars = {}   # {metric: BooleanVar}
        self.invert_vars = {}   # {metric: BooleanVar} for Y-axis inversion
        self.layout_mode = tk.StringVar(value="grid")  # "grid" or "vertical"

        # Default selected metrics (top 4)
        self.default_metrics = [
            'vocabulary_diversity',
            'cloze_score',
            'adjacent_coherence',
            'bigram_repetition_rate'
        ]

        self.setup_ui()

    def setup_ui(self):
        """Setup the GUI layout."""
        # Main container with paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Controls
        left_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(left_frame, weight=0)

        # Right panel - Plots
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        # Setup left panel sections
        self.setup_file_controls(left_frame)
        self.setup_dataset_controls(left_frame)
        self.setup_metric_controls(left_frame)
        self.setup_plot_controls(left_frame)

        # Setup plot area
        self.setup_plot_area(right_frame)

    def setup_file_controls(self, parent):
        """Setup file loading controls."""
        frame = ttk.LabelFrame(parent, text="Data Sources", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            frame,
            text="Add CSV File",
            command=self.add_csv_file
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            frame,
            text="Add Results Directory",
            command=self.add_results_directory
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            frame,
            text="Clear All Data",
            command=self.clear_all_data
        ).pack(fill=tk.X, pady=2)

        # Status label
        self.status_label = ttk.Label(frame, text="No data loaded", foreground="gray")
        self.status_label.pack(fill=tk.X, pady=5)

    def setup_dataset_controls(self, parent):
        """Setup dataset selection controls."""
        frame = ttk.LabelFrame(parent, text="Datasets", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollable frame for dataset checkboxes
        canvas = tk.Canvas(frame, height=150)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.dataset_listframe = ttk.Frame(canvas)

        self.dataset_listframe.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.dataset_listframe, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Select all/none buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            btn_frame,
            text="All",
            command=self.select_all_datasets,
            width=8
        ).pack(side="left", padx=2)

        ttk.Button(
            btn_frame,
            text="None",
            command=self.select_no_datasets,
            width=8
        ).pack(side="left", padx=2)

    def setup_metric_controls(self, parent):
        """Setup metric selection controls."""
        frame = ttk.LabelFrame(parent, text="Metrics", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Scrollable frame for metric checkboxes
        canvas = tk.Canvas(frame, height=200)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.metric_listframe = ttk.Frame(canvas)

        self.metric_listframe.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.metric_listframe, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Create checkboxes for each metric
        for metric in self.available_metrics:
            # Main metric frame
            metric_frame = ttk.Frame(self.metric_listframe)
            metric_frame.pack(anchor="w", pady=2, fill=tk.X)

            # Metric selection checkbox
            var = tk.BooleanVar(value=(metric in self.default_metrics))
            self.metric_vars[metric] = var

            cb = ttk.Checkbutton(
                metric_frame,
                text=self.metric_labels[metric],
                variable=var,
                command=self.update_plot
            )
            cb.pack(anchor="w")

            # Description and invert checkbox on same line
            controls_frame = ttk.Frame(metric_frame)
            controls_frame.pack(anchor="w", fill=tk.X, padx=(15, 0))

            # Add description label
            desc_label = ttk.Label(
                controls_frame,
                text=f"({self.metric_descriptions[metric]})",
                foreground="gray",
                font=("TkDefaultFont", 8)
            )
            desc_label.pack(side="left")

            # Add Y-axis invert checkbox
            invert_var = tk.BooleanVar(value=(metric == 'vocabulary_diversity'))
            self.invert_vars[metric] = invert_var

            invert_cb = ttk.Checkbutton(
                controls_frame,
                text="⇅ Invert",
                variable=invert_var,
                command=self.update_plot
            )
            invert_cb.pack(side="left", padx=(10, 0))
    def _trigger_canvas_resize(self):
        """Force a resize event on the canvas to make figure fit properly."""
        canvas_widget = self.canvas.get_tk_widget()
        # Generate a Configure event with current dimensions
        canvas_widget.event_generate(
            "<Configure>", 
            width=canvas_widget.winfo_width(), 
            height=canvas_widget.winfo_height()
        )

    def setup_plot_controls(self, parent):
        """Setup plot appearance controls."""
        frame = ttk.LabelFrame(parent, text="Plot Controls", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Layout mode control
        layout_frame = ttk.LabelFrame(frame, text="Layout", padding=5)
        layout_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(
            layout_frame,
            text="Grid (2×2, 3×2, etc.)",
            variable=self.layout_mode,
            value="grid",
            command=self.update_plot
        ).pack(anchor="w")

        ttk.Radiobutton(
            layout_frame,
            text="Vertical Stack (full width)",
            variable=self.layout_mode,
            value="vertical",
            command=self.update_plot
        ).pack(anchor="w")

        # DPI control
        dpi_frame = ttk.Frame(frame)
        dpi_frame.pack(fill=tk.X, pady=2)
        ttk.Label(dpi_frame, text="DPI:").pack(side="left")
        self.dpi_var = tk.IntVar(value=100)
        ttk.Spinbox(
            dpi_frame,
            from_=72,
            to=300,
            textvariable=self.dpi_var,
            width=8,
            command=self.update_plot
        ).pack(side="left", padx=5)

        # Update and export buttons
        ttk.Button(
            frame,
            text="Redraw Plot",
            command=self.force_redraw
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            frame,
            text="Export to PNG",
            command=self.export_plot
        ).pack(fill=tk.X, pady=2)


    def force_redraw(self):
        """Force complete window redraw"""
        # Update the canvas widget specifically
        self.canvas.get_tk_widget().update_idletasks()
        
        # Also update the main window if needed
        if hasattr(self, 'master'):
            self.master.update_idletasks()
        elif hasattr(self, 'root'):
            self.root.update_idletasks()


    def setup_plot_area(self, parent):
        """Setup the matplotlib plot area."""
        # Create matplotlib figure with 2x2 subplots
        self.fig = Figure(figsize=(12, 9), dpi=self.dpi_var.get())
        
        self.fig.suptitle('Context Tester - Interactive Analysis', fontsize=14, fontweight='bold')

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()

        # Add toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        # Pack canvas
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def add_csv_file(self):
        """Add a CSV file to the datasets."""
        filetypes = [
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]

        filenames = filedialog.askopenfilenames(
            title="Select CSV files",
            filetypes=filetypes
        )

        if not filenames:
            return

        for filename in filenames:
            try:
                path = Path(filename)
                df, name = load_csv_for_plotting(path)

                if name in self.datasets:
                    name = f"{name}_{len(self.datasets)}"

                self.datasets[name] = df
                self.add_dataset_checkbox(name)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {filename}:\n{str(e)}")
                traceback.print_exc()

        self.update_status()
        self.update_plot()

    def add_results_directory(self):
        """Add a results directory to the datasets."""
        dirname = filedialog.askdirectory(title="Select results directory")

        if not dirname:
            return

        try:
            path = Path(dirname)

            # Check if it's a valid results directory
            if not (path / "metadata.json").exists():
                messagebox.showerror(
                    "Error",
                    f"{dirname} does not appear to be a results directory.\n"
                    "Missing metadata.json file."
                )
                return

            # Load the data
            metadata = load_experiment_metadata(path)
            experiment_meta = metadata.get('experiment_metadata', {})

            model_name = experiment_meta.get('model_name', 'unknown')
            text_name = experiment_meta.get('source_text_name', 'unknown')
            model_id = experiment_meta.get('model_id', '')

            # Strip organization from model name
            if '/' in str(model_name):
                model_name = str(model_name).split('/')[-1]

            # Build name
            parts = [str(model_name), str(text_name)]
            if model_id and str(model_id) != '':
                parts.append(str(model_id))
            name = ' '.join(parts)

            # Load individual rounds and averaged results
            individual_rounds, averaged_results, _ = load_individual_rounds_for_plotting(path)

            # Add averaged dataset
            df_avg = pd.DataFrame(averaged_results)
            df_avg = df_avg[df_avg['context_length'].notna()].copy()

            if name in self.datasets:
                name = f"{name}_{len(self.datasets)}"

            self.datasets[name] = df_avg
            self.add_dataset_checkbox(name)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {dirname}:\n{str(e)}")
            traceback.print_exc()

        self.update_status()
        self.update_plot()

    def add_dataset_checkbox(self, name: str):
        """Add a checkbox for a dataset."""
        if name in self.dataset_vars:
            return

        var = tk.BooleanVar(value=True)
        self.dataset_vars[name] = var

        # Assign color and marker
        colors, markers = self.get_plot_colors_and_markers()
        idx = len(self.dataset_colors)
        self.dataset_colors[name] = colors[idx % len(colors)]
        self.dataset_markers[name] = markers[idx % len(markers)]

        # Create checkbox with color indicator
        frame = ttk.Frame(self.dataset_listframe)
        frame.pack(anchor="w", pady=2, fill=tk.X)

        # Color box
        color_canvas = tk.Canvas(frame, width=20, height=15, highlightthickness=1)
        color_canvas.pack(side="left", padx=(0, 5))
        color_canvas.create_rectangle(
            2, 2, 18, 13,
            fill=self.dataset_colors[name],
            outline="black"
        )

        # Checkbox
        cb = ttk.Checkbutton(
            frame,
            text=name,
            variable=var,
            command=self.update_plot
        )
        cb.pack(side="left", fill=tk.X, expand=True)

    def clear_all_data(self):
        """Clear all loaded datasets."""
        if not self.datasets:
            return

        result = messagebox.askyesno(
            "Confirm",
            "Clear all loaded datasets?"
        )

        if result:
            self.datasets.clear()
            self.dataset_vars.clear()
            self.dataset_colors.clear()
            self.dataset_markers.clear()

            # Clear dataset list
            for widget in self.dataset_listframe.winfo_children():
                widget.destroy()

            self.update_status()
            self.update_plot()

    def select_all_datasets(self):
        """Select all datasets."""
        for var in self.dataset_vars.values():
            var.set(True)
        self.update_plot()

    def select_no_datasets(self):
        """Deselect all datasets."""
        for var in self.dataset_vars.values():
            var.set(False)
        self.update_plot()

    def update_status(self):
        """Update the status label."""
        count = len(self.datasets)
        if count == 0:
            self.status_label.config(text="No data loaded", foreground="gray")
        else:
            self.status_label.config(
                text=f"{count} dataset{'s' if count != 1 else ''} loaded",
                foreground="green"
            )

    def get_plot_colors_and_markers(self):
        """Get colors and markers for plots."""
        colors = [
            '#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea', '#c2410c',
            '#0891b2', '#e11d48', '#65a30d', '#d97706', '#7c3aed', '#ea580c',
            '#0284c7', '#be123c', '#4d7c0f', '#b45309', '#6d28d9', '#c2410c',
            '#0369a1', '#9f1239'
        ]

        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'H', 'X', 'd', 'P', '8']

        return colors, markers

    def update_plot(self):
        """Update the plot based on current selections."""
        try:
            # Get selected datasets
            selected_datasets = [
                name for name, var in self.dataset_vars.items()
                if var.get()
            ]

            # Get selected metrics
            selected_metrics = [
                metric for metric, var in self.metric_vars.items()
                if var.get()
            ]

            # Adjust figure size based on layout and number of metrics
            layout = self.layout_mode.get()
            num_metrics = len(selected_metrics)

            if layout == "vertical" and num_metrics > 0:
                # Vertical layout: increase height proportionally
                # Each plot gets 4 inches of height
                height = max(4 * num_metrics, 6)
                width = 12
            else:
                # Grid layout: standard size
                width, height = 12, 9

            # Update figure size if changed
            current_size = self.fig.get_size_inches()
            if (abs(current_size[0] - width) > 0.1 or
                abs(current_size[1] - height) > 0.1):
                self.fig.set_size_inches(width, height)
                self.canvas.get_tk_widget().config(
                    width=int(width * self.fig.dpi),
                    height=int(height * self.fig.dpi)
                )

            # Clear figure
            self.fig.clear()

            if not selected_datasets:
                self.fig.text(
                    0.5, 0.5,
                    'No datasets selected\n\nAdd data using the controls on the left',
                    ha='center', va='center',
                    fontsize=14, color='gray'
                )
                self.canvas.draw_idle()  # Changed to draw_idle
                self.canvas.get_tk_widget().update_idletasks()
                return

            if not selected_metrics:
                self.fig.text(
                    0.5, 0.5,
                    'No metrics selected\n\nSelect metrics to plot',
                    ha='center', va='center',
                    fontsize=14, color='gray'
                )
                self.canvas.draw_idle()  # Changed to draw_idle
                self.canvas.get_tk_widget().update_idletasks()
                return

            # Determine subplot layout based on mode
            num_metrics = len(selected_metrics)
            layout = self.layout_mode.get()

            if layout == "vertical":
                # Vertical stacking - each plot gets full width
                rows, cols = num_metrics, 1
            else:
                # Grid layout
                if num_metrics == 1:
                    rows, cols = 1, 1
                elif num_metrics == 2:
                    rows, cols = 1, 2
                elif num_metrics <= 4:
                    rows, cols = 2, 2
                elif num_metrics <= 6:
                    rows, cols = 2, 3
                else:
                    rows, cols = 3, 3

            # Update figure title
            self.fig.suptitle('Context Tester - Interactive Analysis', fontsize=14, fontweight='bold')

            # Create subplots
            for idx, metric in enumerate(selected_metrics):
                ax = self.fig.add_subplot(rows, cols, idx + 1)

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

                    # Plot
                    ax.plot(
                        x_data, y_data,
                        marker=self.dataset_markers[dataset_name],
                        color=self.dataset_colors[dataset_name],
                        linewidth=2,
                        markersize=6,
                        label=dataset_name,
                        linestyle='-'
                    )

                # Format subplot
                ax.set_xscale('log')
                ax.set_xlabel('Context Length', fontsize=10)
                ax.set_ylabel(self.metric_labels[metric], fontsize=10)
                ax.set_title(
                    f"{self.metric_labels[metric]}\n({self.metric_descriptions[metric]})",
                    fontsize=10,
                    fontweight='bold'
                )
                ax.grid(True, alpha=0.3)

                # Invert Y-axis if requested by user
                if metric in self.invert_vars and self.invert_vars[metric].get():
                    ax.invert_yaxis()

                # Format x-axis labels
                if len(self.datasets) > 0:
                    # Get all unique context lengths across selected datasets
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
                            fontsize=6
                        )

                # Add legend (only if multiple datasets)
                if len(selected_datasets) > 1:
                    ax.legend(loc='best', fontsize=7)

           
            self.fig.tight_layout()
            self.canvas.draw_idle()
            
            # Force a resize event to make figure fit canvas
            self._trigger_canvas_resize()
            
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to update plot:\n{str(e)}")
            traceback.print_exc()
            
    def _on_canvas_resize(self, event):
        """Handle canvas resize events."""
        if event.width > 1 and event.height > 1:  # Avoid initial small values
            new_width_inches = event.width / self.fig.dpi
            new_height_inches = event.height / self.fig.dpi
            self.fig.set_size_inches(new_width_inches, new_height_inches)
            self.canvas.draw_idle()

    def export_plot(self):
        """Export the current plot to PNG."""
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
            dpi = self.dpi_var.get()
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            messagebox.showinfo("Success", f"Plot exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{str(e)}")


def main():
    """Main entry point for the GUI."""
    root = tk.Tk()
    app = PlotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
