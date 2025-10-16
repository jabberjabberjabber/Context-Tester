#!/usr/bin/env python3
"""
GUI for Context Tester Benchmark Tool

Provides a user-friendly interface for configuring and running benchmarks.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
import sys
import os
import json
from datetime import datetime

# Import parameter schema
from src.parameter_schema import PARAMETER_SCHEMA, get_gui_params_by_section, get_cli_name, get_var_name

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('TkAgg')


class BenchmarkGUI:
    """GUI for running context tester benchmarks."""

    def __init__(self, root):
        self.root = root
        self.root.title("Context Tester - Benchmark Tool")
        self.root.geometry("1080x900")

        # Settings file
        self.settings_file = Path.home() / ".context_tester_gui_settings.json"

        # Configuration variables (auto-generated from parameter_schema.py)
        self.text_file = tk.StringVar()

        # Dynamically create StringVars for all parameters in schema
        for param_name, spec in PARAMETER_SCHEMA.items():
            if not spec.get('gui', True) or spec.get('type') == 'positional':
                continue

            var_name = get_var_name(param_name)
            default = spec.get('default')

            if default is not None:
                setattr(self, var_name, tk.StringVar(value=str(default)))
            else:
                setattr(self, var_name, tk.StringVar())

        # Runtime state
        self.running = False
        self.benchmark_thread = None
        self.recent_hosts = []
        self.available_models = []

        # Load settings first
        self.load_settings()

        self.setup_ui()

        # Apply environment variables after UI is created (so we can update the label)
        self.apply_env_variables()

        # Save settings on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        """Setup the main UI layout."""
        # Create main container with scrollbar
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side: Configuration
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Right side: Output console
        right_frame = ttk.Frame(main_container)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Setup sections
        self.setup_text_file_section(left_frame)
        self.setup_api_section(left_frame)
        self.setup_test_params_section(left_frame)
        self.setup_generation_params_section(left_frame)
        self.setup_run_controls(left_frame)
        self.setup_output_console(right_frame)

    def setup_text_file_section(self, parent):
        """Setup text file selection section."""
        frame = ttk.LabelFrame(parent, text="Input Text", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        # File selection
        file_frame = ttk.Frame(frame)
        file_frame.pack(fill=tk.X)

        ttk.Label(file_frame, text="Text File:").pack(side=tk.LEFT, padx=(0, 5))

        entry = ttk.Entry(file_frame, textvariable=self.text_file, width=10)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        ttk.Button(
            file_frame,
            text="Browse...",
            command=self.browse_text_file
        ).pack(side=tk.LEFT)

    def setup_api_section(self, parent):
        """Setup API configuration section."""
        frame = ttk.LabelFrame(parent, text="API Configuration", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        # API URL with dropdown
        url_frame = ttk.Frame(frame)
        url_frame.pack(fill=tk.X, pady=2)
        ttk.Label(url_frame, text="API URL:", width=15).pack(side=tk.LEFT)

        # Combobox for recent hosts
        self.url_combo = ttk.Combobox(url_frame, textvariable=self.api_url)
        self.url_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.url_combo['values'] = self.recent_hosts

        # Detect API button
        ttk.Button(
            url_frame,
            text="Detect",
            command=self.detect_api_type,
            width=8
        ).pack(side=tk.LEFT)

        # API Password/Key
        pwd_frame = ttk.Frame(frame)
        pwd_frame.pack(fill=tk.X, pady=2)
        ttk.Label(pwd_frame, text="API Key:", width=15).pack(side=tk.LEFT)
        ttk.Entry(pwd_frame, textvariable=self.api_password, show="*").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        # Environment variable indicator
        self.env_key_label = ttk.Label(pwd_frame, text="", foreground="green", font=('TkDefaultFont', 8))
        self.env_key_label.pack(side=tk.LEFT)

        # Model Name with combobox and fetch button
        model_frame = ttk.Frame(frame)
        model_frame.pack(fill=tk.X, pady=2)
        ttk.Label(model_frame, text="Model Name:", width=15).pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_name)
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.model_combo['values'] = self.available_models

        # Fetch models button
        ttk.Button(
            model_frame,
            text="Fetch",
            command=self.fetch_models,
            width=8
        ).pack(side=tk.LEFT)

        # Tokenizer Model with combobox
        tokenizer_frame = ttk.Frame(frame)
        tokenizer_frame.pack(fill=tk.X, pady=2)
        ttk.Label(tokenizer_frame, text="Tokenizer Model:", width=15).pack(side=tk.LEFT)
        self.tokenizer_combo = ttk.Combobox(tokenizer_frame, textvariable=self.tokenizer_model)
        self.tokenizer_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.tokenizer_combo['values'] = self.available_models

        # Help text
        self.api_help_label = ttk.Label(
            frame,
            text="Click 'Detect' to auto-configure for KoboldCpp\n"
                 "For other APIs: Set Model Name and Tokenizer Model manually\n"
                 "Tokenizer is Huggingface Tokenizer name and may not match!",
            font=('TkDefaultFont', 8),
            foreground='gray')
        self.api_help_label.pack(fill=tk.X, pady=(5, 0))

    def setup_test_params_section(self, parent):
        """Setup test parameters section (auto-generated from parameter_schema)."""
        frame = ttk.LabelFrame(parent, text="Test Parameters", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        # Dynamically generate fields from schema
        params = get_gui_params_by_section('test')
        for param_name, spec in params.items():
            var_name = get_var_name(param_name)
            gui_label = spec.get('gui_label', param_name.replace('_', ' ').title())
            gui_type = spec.get('gui_type', 'entry')
            gui_width = spec.get('gui_width')
            gui_hint = spec.get('gui_hint')

            # Create frame for this field
            field_frame = ttk.Frame(frame)
            field_frame.pack(fill=tk.X, pady=2)
            ttk.Label(field_frame, text=f"{gui_label}:", width=15).pack(side=tk.LEFT)

            # Create widget based on type
            if gui_type == 'checkbox':
                ttk.Checkbutton(field_frame, variable=getattr(self, var_name)).pack(side=tk.LEFT)
            elif gui_type == 'combobox':
                combo = ttk.Combobox(field_frame, textvariable=getattr(self, var_name))
                # Set dropdown values if specified
                if spec.get('gui_values'):
                    combo['values'] = spec['gui_values']
                if gui_width:
                    combo.pack(side=tk.LEFT, fill=tk.X, expand=False)
                else:
                    combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
            else:  # entry
                if gui_width:
                    ttk.Entry(field_frame, textvariable=getattr(self, var_name), width=gui_width).pack(side=tk.LEFT)
                else:
                    ttk.Entry(field_frame, textvariable=getattr(self, var_name)).pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Add hint text if provided
            if gui_hint:
                ttk.Label(field_frame, text=gui_hint, font=('TkDefaultFont', 8), foreground='gray').pack(side=tk.LEFT, padx=(5, 0))

    def setup_generation_params_section(self, parent):
        """Setup generation parameters section (auto-generated from parameter_schema)."""
        frame = ttk.LabelFrame(parent, text="Generation Parameters", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        # Dynamically generate fields from schema
        params = get_gui_params_by_section('generation')
        for param_name, spec in params.items():
            var_name = get_var_name(param_name)
            gui_label = spec.get('gui_label', param_name.replace('_', ' ').title())
            gui_type = spec.get('gui_type', 'entry')
            gui_width = spec.get('gui_width')
            gui_hint = spec.get('gui_hint')

            # Create frame for this field
            field_frame = ttk.Frame(frame)
            field_frame.pack(fill=tk.X, pady=2)
            ttk.Label(field_frame, text=f"{gui_label}:", width=15).pack(side=tk.LEFT)

            # Create widget based on type
            if gui_type == 'checkbox':
                ttk.Checkbutton(field_frame, variable=getattr(self, var_name)).pack(side=tk.LEFT)
            elif gui_type == 'combobox':
                combo = ttk.Combobox(field_frame, textvariable=getattr(self, var_name))
                # Set dropdown values if specified
                if spec.get('gui_values'):
                    combo['values'] = spec['gui_values']
                if gui_width:
                    combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
                else:
                    combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
            else:  # entry
                if gui_width:
                    ttk.Entry(field_frame, textvariable=getattr(self, var_name), width=gui_width).pack(side=tk.LEFT)
                else:
                    ttk.Entry(field_frame, textvariable=getattr(self, var_name)).pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Add hint text if provided
            if gui_hint:
                ttk.Label(field_frame, text=gui_hint, font=('TkDefaultFont', 8), foreground='gray').pack(side=tk.LEFT, padx=(5, 0))

    def setup_run_controls(self, parent):
        """Setup run button and controls."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(0, 10))

        self.run_button = ttk.Button(
            frame,
            text="Start Benchmark",
            command=self.start_benchmark,
            style='Accent.TButton'
        )
        self.run_button.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_button = ttk.Button(
            frame,
            text="Stop",
            command=self.stop_benchmark,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT)

        # Status label
        self.status_label = ttk.Label(frame, text="Ready", foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))

    def setup_output_console(self, parent):
        """Setup output console for real-time feedback."""
        frame = ttk.LabelFrame(parent, text="Output Console", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Scrolled text widget
        self.console = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            width=60,
            height=40,
            font=('Courier', 9),
            bg='#1e1e1e',
            fg='#d4d4d4',
            insertbackground='white'
        )
        self.console.pack(fill=tk.BOTH, expand=True)

        # Configure tags for colored output
        self.console.tag_config('info', foreground='#4ec9b0')
        self.console.tag_config('warning', foreground='#dcdcaa')
        self.console.tag_config('error', foreground='#f48771')
        self.console.tag_config('success', foreground='#4fc1ff')

        # Make read-only
        self.console.config(state=tk.DISABLED)

        # Clear button
        ttk.Button(
            frame,
            text="Clear Console",
            command=self.clear_console
        ).pack(pady=(5, 0))

    def browse_text_file(self):
        """Browse for text file."""
        filename = filedialog.askopenfilename(
            title="Select text file",
            filetypes=[
                ("Text files", "*.txt"),
                ("PDF files", "*.pdf"),
                ("HTML files", "*.html *.htm"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.text_file.set(filename)

    def fetch_models(self):
        """Fetch available models from the API endpoint."""
        if not self.api_url.get():
            messagebox.showwarning("No URL", "Please enter an API URL first")
            return

        self.log_to_console("Fetching available models...", 'info')

        try:
            import requests

            base_url = self.api_url.get().replace('/v1/chat/completions', '').rstrip('/')
            models_url = f"{base_url}/v1/models"

            self.log_to_console(f"Querying: {models_url}", 'info')

            headers = {}
            if self.api_password.get():
                headers['Authorization'] = f"Bearer {self.api_password.get()}"

            response = requests.get(models_url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Extract model IDs
                models = []
                if 'data' in data:
                    for model in data['data']:
                        if 'id' in model:
                            models.append(model['id'])
                elif 'models' in data:
                    models = data['models']
                elif isinstance(data, list):
                    models = [m.get('id', m) for m in data if isinstance(m, dict)]

                if models:
                    self.available_models = sorted(models)
                    self.model_combo['values'] = self.available_models
                    self.tokenizer_combo['values'] = self.available_models

                    self.log_to_console(f"✓ Found {len(models)} models", 'success')

                    # Show in dialog
                    model_list = '\n'.join(models[:20])  # Show first 20
                    if len(models) > 20:
                        model_list += f"\n... and {len(models) - 20} more"

                    messagebox.showinfo(
                        "Models Found",
                        f"Found {len(models)} available models:\n\n{model_list}\n\n"
                        f"Select from dropdown or type to search."
                    )
                else:
                    self.log_to_console("No models found in response", 'warning')
                    messagebox.showwarning(
                        "No Models",
                        "API responded but no models were found.\n\n"
                        "You may need to manually enter the model name."
                    )
            else:
                self.log_to_console(f"Failed: HTTP {response.status_code}", 'error')
                messagebox.showerror(
                    "Fetch Failed",
                    f"Could not fetch models.\n\n"
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )

        except requests.exceptions.Timeout:
            self.log_to_console("Request timed out", 'error')
            messagebox.showerror("Timeout", "Request timed out. Check your API URL and connection.")
        except Exception as e:
            self.log_to_console(f"Error fetching models: {str(e)}", 'error')
            import traceback
            self.log_to_console(traceback.format_exc(), 'error')
            messagebox.showerror("Error", f"Could not fetch models:\n{str(e)}")

    def detect_api_type(self):
        """Detect if the API is KoboldCpp and update help text."""
        if not self.api_url.get():
            messagebox.showwarning("No URL", "Please enter an API URL first")
            return

        self.log_to_console("Detecting API type...", 'info')

        try:
            import requests
            import json

            # Detect KoboldCpp without creating full client (to avoid tokenizer issues)
            base_url = self.api_url.get().replace('/v1/chat/completions', '').rstrip('/')
            version_url = f"{base_url}/api/extra/version"

            self.log_to_console(f"Checking: {version_url}", 'info')

            is_kobold = False
            try:
                response = requests.get(version_url, timeout=5)
                if response.status_code == 200:
                    version_data = response.json()
                    # Check if response has KoboldCpp-specific fields
                    if 'result' in version_data or 'version' in version_data:
                        is_kobold = True
                        self.log_to_console(f"KoboldCpp version info: {version_data}", 'info')
            except requests.exceptions.Timeout:
                self.log_to_console("Request timed out - assuming not KoboldCpp", 'warning')
            except requests.exceptions.RequestException as e:
                self.log_to_console(f"Connection failed: {str(e)} - assuming not KoboldCpp", 'warning')

            if is_kobold:
                self.log_to_console("✓ Detected KoboldCpp API", 'success')
                self.api_help_label.config(
                    text="✓ KoboldCpp detected - Model Name and Tokenizer will be auto-detected",
                    foreground='green'
                )
                # Clear model fields since they'll be auto-detected
                self.model_name.set("")
                self.tokenizer_model.set("")

                messagebox.showinfo(
                    "KoboldCpp Detected",
                    "This is a KoboldCpp API.\n\n"
                    "Model Name and Tokenizer fields have been cleared.\n"
                    "They will be auto-detected when you run the benchmark."
                )
            else:
                self.log_to_console("✗ Not a KoboldCpp API (likely NVIDIA NIM or other)", 'warning')
                self.api_help_label.config(
                    text="Not KoboldCpp - Please set Model Name and Tokenizer Model manually",
                    foreground='orange'
                )

                messagebox.showinfo(
                    "Not KoboldCpp",
                    "This does not appear to be a KoboldCpp API.\n\n"
                    "Please manually configure:\n"
                    "- Model Name\n"
                    "- Tokenizer Model\n\n"
                    "For NVIDIA NIM, check nim_models.md for known working configurations."
                )

        except Exception as e:
            self.log_to_console(f"Error detecting API: {str(e)}", 'error')
            import traceback
            self.log_to_console(traceback.format_exc(), 'error')
            messagebox.showerror("Detection Error", f"Could not detect API type:\n{str(e)}")

    def log_to_console(self, message, tag='info'):
        """Log message to console with optional color tag."""
        self.console.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.insert(tk.END, f"[{timestamp}] ", 'info')
        self.console.insert(tk.END, f"{message}\n", tag)
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)

    def clear_console(self):
        """Clear the console."""
        self.console.config(state=tk.NORMAL)
        self.console.delete(1.0, tk.END)
        self.console.config(state=tk.DISABLED)

    def validate_inputs(self):
        """Validate user inputs before running benchmark."""
        errors = []

        if not self.text_file.get():
            errors.append("Please select a text file")
        elif not Path(self.text_file.get()).exists():
            errors.append("Text file does not exist")

        if not self.api_url.get():
            errors.append("API URL is required")

        # Check if tokenizer/model are required (not KoboldCpp)
        if self.api_url.get():
            try:
                import requests
                import json

                # Quick check for KoboldCpp without creating full client
                base_url = self.api_url.get().replace('/v1/chat/completions', '').rstrip('/')
                version_url = f"{base_url}/api/extra/version"

                is_kobold = False
                try:
                    response = requests.get(version_url, timeout=3)
                    if response.status_code == 200:
                        version_data = response.json()
                        if 'result' in version_data or 'version' in version_data:
                            is_kobold = True
                except:
                    pass

                # If not KoboldCpp, require model name and tokenizer
                if not is_kobold:
                    if not self.model_name.get():
                        errors.append("Model Name is required for non-KoboldCpp APIs\n(Click 'Detect' to check API type)")
                    if not self.tokenizer_model.get():
                        errors.append("Tokenizer Model is required for non-KoboldCpp APIs\n(Click 'Detect' to check API type)")

            except Exception:
                # If detection fails, warn user only if fields are empty
                if not self.model_name.get() and not self.tokenizer_model.get():
                    errors.append("Could not detect API type. For non-KoboldCpp APIs,\nplease provide Model Name and Tokenizer Model")

        try:
            max_ctx = int(self.max_context.get())
            if max_ctx <= 0:
                errors.append("Max context must be positive")
        except ValueError:
            errors.append("Max context must be a number")

        try:
            rounds = int(self.rounds.get())
            if rounds <= 0:
                errors.append("Rounds must be positive")
        except ValueError:
            errors.append("Rounds must be a number")

        try:
            max_tokens = int(self.max_tokens.get())
            if max_tokens <= 0:
                errors.append("Max tokens must be positive")
        except ValueError:
            errors.append("Max tokens must be a number")

        try:
            temp = float(self.temperature.get())
            if temp < 0:
                errors.append("Temperature must be non-negative")
        except ValueError:
            errors.append("Temperature must be a number")

        return errors

    def build_command_args(self):
        """Build command-line arguments (auto-generated from parameter_schema)."""
        args = [self.text_file.get()]

        # Dynamically build arguments from schema
        for param_name, spec in PARAMETER_SCHEMA.items():
            if not spec.get('gui', True) or spec.get('type') == 'positional':
                continue

            var_name = get_var_name(param_name)
            cli_name = get_cli_name(param_name, spec)
            value = getattr(self, var_name).get()

            if spec['type'] == 'bool':
                # Boolean flags
                if value == 'True':
                    args.append(f'--{cli_name}')
            elif param_name == 'api_password':
                # Special handling for API password
                if value and not self._is_from_env_variable():
                    args.extend([f'--{cli_name}', value])
            else:
                # Regular parameters
                if value:
                    args.extend([f'--{cli_name}', value])

        return args

    def start_benchmark(self):
        """Start the benchmark in a separate thread."""
        # Validate inputs
        errors = self.validate_inputs()
        if errors:
            # Log to console as well
            for error in errors:
                self.log_to_console(f"Validation Error: {error}", 'error')

            messagebox.showerror("Validation Error", "\n".join(errors))
            return

        # Update UI state
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Running...", foreground="green")

        # Clear console
        self.clear_console()
        self.log_to_console("Starting benchmark...", 'success')

        # Build command
        args = self.build_command_args()
        self.log_to_console(f"Command: main.py {' '.join(args)}", 'info')

        # Run in thread
        self.benchmark_thread = threading.Thread(target=self.run_benchmark_thread, args=(args,))
        self.benchmark_thread.daemon = True
        self.benchmark_thread.start()

    def run_benchmark_thread(self, args):
        """Run the benchmark in a background thread."""
        try:
            # Import main benchmark module
            from src.config import parse_args
            from main import run_benchmark

            # Redirect stdout to console
            import io

            class ConsoleRedirector:
                def __init__(self, gui, tag='info'):
                    self.gui = gui
                    self.tag = tag

                def write(self, message):
                    if message.strip():
                        self.gui.root.after(0, lambda: self.gui.log_to_console(message.strip(), self.tag))

                def flush(self):
                    pass

            old_stdout = sys.stdout
            old_stderr = sys.stderr

            sys.stdout = ConsoleRedirector(self, 'info')
            sys.stderr = ConsoleRedirector(self, 'error')

            # Parse arguments and run
            parsed_args = parse_args(args)
            run_benchmark(parsed_args)

            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Success
            self.root.after(0, lambda: self.log_to_console("Benchmark completed successfully!", 'success'))

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self.root.after(0, lambda: self.log_to_console(f"Error: {str(e)}", 'error'))
            self.root.after(0, lambda: self.log_to_console(error_msg, 'error'))

        finally:
            # Reset UI state
            self.root.after(0, self.benchmark_finished)

    def stop_benchmark(self):
        """Stop the running benchmark."""
        if self.running:
            self.log_to_console("Stopping benchmark (may take a moment)...", 'warning')
            self.running = False
            # Note: Actual stopping mechanism would need to be implemented in main.py

    def benchmark_finished(self):
        """Called when benchmark finishes."""
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Ready", foreground="gray")

    def load_settings(self):
        """Load settings from JSON file (auto-generated from parameter_schema)."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)

                # Load recent hosts
                self.recent_hosts = settings.get('recent_hosts', [])

                # Dynamically load last used values from schema
                last_values = settings.get('last_values', {})
                for param_name, spec in PARAMETER_SCHEMA.items():
                    if not spec.get('gui', True) or spec.get('type') == 'positional':
                        continue

                    var_name = get_var_name(param_name)
                    if last_values.get(var_name):
                        getattr(self, var_name).set(last_values[var_name])

                print(f"Loaded settings from {self.settings_file}")

        except Exception as e:
            print(f"Could not load settings: {e}")

    def save_settings(self):
        """Save settings to JSON file (auto-generated from parameter_schema)."""
        try:
            # Update recent hosts
            current_url = self.api_url.get()
            if current_url and current_url not in self.recent_hosts:
                self.recent_hosts.insert(0, current_url)
                # Keep only last 10 hosts
                self.recent_hosts = self.recent_hosts[:10]
            elif current_url in self.recent_hosts:
                # Move to front
                self.recent_hosts.remove(current_url)
                self.recent_hosts.insert(0, current_url)

            # Dynamically build last_values from schema
            last_values = {}
            for param_name, spec in PARAMETER_SCHEMA.items():
                if not spec.get('gui', True) or spec.get('type') == 'positional':
                    continue

                # Don't save API password to disk
                if param_name == 'api_password':
                    continue

                var_name = get_var_name(param_name)
                last_values[var_name] = getattr(self, var_name).get()

            settings = {
                'recent_hosts': self.recent_hosts,
                'last_values': last_values
            }

            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)

            print(f"Saved settings to {self.settings_file}")

        except Exception as e:
            print(f"Could not save settings: {e}")

    def apply_env_variables(self):
        """Apply API key from environment variables if available."""
        if not self.api_password.get():
            # Check for API key in environment
            api_key = os.environ.get('API_KEY') or \
                     os.environ.get('API_PASSWORD') or \
                     os.environ.get('OPENAI_API_KEY') or \
                     os.environ.get('NVIDIA_API_KEY') or \
                     os.environ.get('NVAPI_KEY')

            if api_key:
                self.api_password.set(api_key)
                self._env_api_key = api_key  # Remember it came from env
                print(f"Loaded API key from environment variable")
                # Update indicator after UI is created
                if hasattr(self, 'env_key_label'):
                    self.env_key_label.config(text="✓ from env")
            else:
                self._env_api_key = None
        else:
            self._env_api_key = None

    def _is_from_env_variable(self):
        """Check if current API password came from environment variable."""
        return hasattr(self, '_env_api_key') and \
               self._env_api_key and \
               self.api_password.get() == self._env_api_key

    def on_closing(self):
        """Handle window close event."""
        self.save_settings()
        self.root.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()
    app = BenchmarkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
