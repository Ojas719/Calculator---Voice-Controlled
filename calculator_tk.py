"""Tkinter desktop calculator using a safe evaluator.

Run interactive GUI:
    python "c:\\Users\\OJAS RANA\\OneDrive\\Documents\\calculator_tk.py"

Run self-tests (non-GUI):
    python "c:\\Users\\OJAS RANA\\OneDrive\\Documents\\calculator_tk.py" --test
"""

import ast
import math
import operator as op
import re
import sys
from io import BytesIO
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:
    tk = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation
except Exception:
    plt = None
    np = None
    FuncAnimation = None


class SafeEvaluator(ast.NodeVisitor):
    ALLOWED_BINOPS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
    }
    ALLOWED_UNARYOPS = {ast.UAdd: op.pos, ast.USub: op.neg}

    def __init__(self, names=None, funcs=None):
        self.names = {} if names is None else dict(names)
        self.funcs = {} if funcs is None else dict(funcs)

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        return getattr(self, method, self.generic_visit)(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError('Only numeric constants allowed')

    def visit_Num(self, node):
        return node.n

    def visit_BinOp(self, node):
        op_type = type(node.op)
        if op_type not in self.ALLOWED_BINOPS:
            raise ValueError('Operation not allowed')
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self.ALLOWED_BINOPS[op_type](left, right)

    def visit_UnaryOp(self, node):
        op_type = type(node.op)
        if op_type not in self.ALLOWED_UNARYOPS:
            raise ValueError('Unary op not allowed')
        return self.ALLOWED_UNARYOPS[op_type](self.visit(node.operand))

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError('Only direct function calls allowed')
        fname = node.func.id
        if fname not in self.funcs:
            raise ValueError(f'Function {fname} not allowed')
        args = [self.visit(a) for a in node.args]
        return self.funcs[fname](*args)

    def visit_Name(self, node):
        if node.id in self.names:
            return self.names[node.id]
        raise ValueError(f'Unknown identifier: {node.id}')

    def generic_visit(self, node):
        raise ValueError(f'Unsupported: {node.__class__.__name__}')


def preprocess(expr: str) -> str:
    """Convert human language math expressions to Python syntax."""
    expr = expr.strip()
    expr_lower = expr.lower()
    
    # Handle natural language to math conversions
    # Power operations
    expr = re.sub(r'(\w+)\s+squared', r'\1**2', expr, flags=re.IGNORECASE)
    expr = re.sub(r'(\w+)\s+square\b', r'\1**2', expr, flags=re.IGNORECASE)
    expr = re.sub(r'(\w+)\s+cubed', r'\1**3', expr, flags=re.IGNORECASE)
    expr = re.sub(r'(\w+)\s+cube\b', r'\1**3', expr, flags=re.IGNORECASE)
    expr = re.sub(r'(\)|\w+)\s+to\s+the\s+power\s+of\s+(\d+)', r'\1**\2', expr, flags=re.IGNORECASE)
    expr = re.sub(r'(\)|\w+)\s+power\s+(\d+)', r'\1**\2', expr, flags=re.IGNORECASE)
    expr = re.sub(r'(\)|\w+)\s+raised\s+to\s+(\d+)', r'\1**\2', expr, flags=re.IGNORECASE)
    
    # Roots and special functions
    expr = re.sub(r'\bsquare\s+root\s+of\b', 'sqrt(', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bsquare\s+root', 'sqrt', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bcube\s+root\s+of\b', 'cbrt(', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\broot\s+of\b', 'sqrt(', expr, flags=re.IGNORECASE)
    
    # Trigonometric functions
    expr = re.sub(r'\bsine\b', 'sin', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bcosine\b', 'cos', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\btangent\b', 'tan', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\barcsine\b', 'asin', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\barccosine\b', 'acos', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\barctangent\b', 'atan', expr, flags=re.IGNORECASE)
    
    # Logarithmic functions - handle all variations
    expr = re.sub(r'\bnatural\s+log\b', 'log', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\blog\s+base\s+10\b', 'log10', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\blog\s+10\b', 'log10', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bln\s*\(', 'log(', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bln\b', 'log', expr, flags=re.IGNORECASE)
    
    # Exponential
    expr = re.sub(r'\be\s+raised\s+to\b', 'exp', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\be\s+power\b', 'exp', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bexp\s+of\b', 'exp(', expr, flags=re.IGNORECASE)
    
    # Arithmetic operators
    expr = re.sub(r'\btimes\b', '*', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bmultiply\b', '*', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bdivide\s+by\b', '/', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bdivided\s+by\b', '/', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bplus\b', '+', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bminus\b', '-', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bsubtract\b', '-', expr, flags=re.IGNORECASE)
    
    # Constants
    expr = re.sub(r'\bpi\b', 'pi', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\beuler(\s+number)?\b', 'e', expr, flags=re.IGNORECASE)
    
    # Factorial
    expr = re.sub(r'(?P<val>(?:\d+\.?\d*|\([^()]*\)))\s*factorial', r'\g<val>!', expr, flags=re.IGNORECASE)
    expr = re.sub(r'(?P<val>(?:\d+\.?\d*|\([^()]*\)))\s*!', r'factorial(\g<val>)', expr)
    
    # Handle closing parentheses for functions like "sqrt("
    expr = re.sub(r'sqrt\s*\(\s*', 'sqrt(', expr, flags=re.IGNORECASE)
    expr = re.sub(r'sin\s*\(\s*', 'sin(', expr, flags=re.IGNORECASE)
    expr = re.sub(r'cos\s*\(\s*', 'cos(', expr, flags=re.IGNORECASE)
    expr = re.sub(r'tan\s*\(\s*', 'tan(', expr, flags=re.IGNORECASE)
    expr = re.sub(r'log\s*\(\s*', 'log(', expr, flags=re.IGNORECASE)
    expr = re.sub(r'log10\s*\(\s*', 'log10(', expr, flags=re.IGNORECASE)
    expr = re.sub(r'exp\s*\(\s*', 'exp(', expr, flags=re.IGNORECASE)
    
    # Handle "of" before closing parentheses for functions
    expr = re.sub(r'(\w+)\s+of\s+', r'\1(', expr, flags=re.IGNORECASE)
    
    # Standard symbol replacements
    expr = expr.replace('^', '**')
    
    return expr


def auto_detect_graph_limits(expr: str, initial_range=20, max_attempts=3):
    """Automatically detect appropriate graph limits based on expression behavior."""
    expr_processed = preprocess(expr)
    names, funcs = build_context()
    
    attempt = 0
    while attempt < max_attempts:
        x_min = -initial_range
        x_max = initial_range
        
        try:
            x_vals = np.linspace(x_min, x_max, 200)
            y_vals = []
            
            for x_val in x_vals:
                try:
                    eval_names = names.copy()
                    eval_names['x'] = float(x_val)
                    tree = ast.parse(expr_processed, mode='eval')
                    evaluator = SafeEvaluator(names=eval_names, funcs=funcs)
                    y_val = evaluator.visit(tree)
                    
                    if np.isfinite(y_val):
                        y_vals.append(float(y_val))
                    else:
                        y_vals.append(np.nan)
                except:
                    y_vals.append(np.nan)
            
            y_vals = np.array(y_vals)
            
            # Check if we have valid data
            valid_y = y_vals[~np.isnan(y_vals)]
            if len(valid_y) > 0:
                y_min, y_max = np.min(valid_y), np.max(valid_y)
                y_range = y_max - y_min
                
                # If range is too large, reduce x range
                if y_range > 1000:
                    initial_range = initial_range / 2
                    attempt += 1
                    continue
                
                return x_min, x_max
            else:
                # No valid values, expand range
                initial_range = initial_range * 1.5
                attempt += 1
                continue
        except:
            initial_range = initial_range * 1.5
            attempt += 1
            continue
    
    # Fallback
    return -10, 10


def graph_expression(expr: str, x_min=None, x_max=None, title='Function Graph'):
    """Generate and display an animated, interactive graph for a mathematical expression."""
    if plt is None or np is None:
        raise RuntimeError('matplotlib and numpy not installed. Run: pip install matplotlib numpy')
    
    # Auto-detect limits if not provided
    if x_min is None or x_max is None:
        x_min, x_max = auto_detect_graph_limits(expr)
    
    expr_processed = preprocess(expr)
    
    try:
        # Get context
        names, funcs = build_context()
        
        # Test parse the expression first
        ast.parse(expr_processed, mode='eval')
        
        # Generate x values
        x_vals = np.linspace(x_min, x_max, 300)
        y_vals = []
        
        # Evaluate expression for each x value
        for x_val in x_vals:
            try:
                # Create fresh names dict with current x value
                eval_names = names.copy()
                eval_names['x'] = float(x_val)
                
                tree = ast.parse(expr_processed, mode='eval')
                evaluator = SafeEvaluator(names=eval_names, funcs=funcs)
                y_val = evaluator.visit(tree)
                
                # Check if result is a valid number
                if np.isfinite(y_val):
                    y_vals.append(float(y_val))
                else:
                    y_vals.append(np.nan)
            except:
                y_vals.append(np.nan)
        
        y_vals = np.array(y_vals)
        
        # Create figure with dark theme
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#163e39')
        
        # Set limits
        ax.set_xlim(x_min, x_max)
        y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
        y_range = y_max - y_min if y_max > y_min else 10
        ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        # Grid and styling
        ax.grid(True, color='#0f3460', linestyle='--', alpha=0.5)
        ax.set_xlabel('x', fontsize=12, color='#eee', fontweight='bold')
        ax.set_ylabel('f(x)', fontsize=12, color='#eee', fontweight='bold')
        ax.set_title(title, fontsize=14, color='#2ECC71', fontweight='bold', pad=20)
        ax.tick_params(colors='#eee')
        
        # Spines
        for spine in ax.spines.values():
            spine.set_color('#FF8C00')
            spine.set_linewidth(2)
        
        # Create line object for animation
        line, = ax.plot([], [], color='#2ECC71', linewidth=2.5, label=expr)
        
        # Hover text for coordinates
        hover_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', 
                            facecolor='#0f3460', alpha=0.8, edgecolor='#2ECC71'),
                            color='#2ECC71', family='monospace')
        hover_text.set_visible(False)
        
        # Store original limits for zoom/pan
        class InteractionState:
            def __init__(self):
                self.x_min = x_min
                self.x_max = x_max
                self.y_min = y_min - y_range * 0.1
                self.y_max = y_max + y_range * 0.1
        
        state = InteractionState()
        
        # Animation function
        def animate(frame):
            # Draw progressively more points
            end_idx = int((frame / 100) * len(x_vals))
            if end_idx > 0:
                # Filter out NaN values for smooth drawing
                x_plot = x_vals[:end_idx]
                y_plot = y_vals[:end_idx]
                
                # Remove NaN sequences for cleaner animation
                valid_mask = ~np.isnan(y_plot)
                if np.any(valid_mask):
                    x_plot = x_plot[valid_mask]
                    y_plot = y_plot[valid_mask]
                    line.set_data(x_plot, y_plot)
            
            return line,
        
        # Mouse hover to show coordinates
        def on_move(event):
            if event.inaxes != ax:
                hover_text.set_visible(False)
                fig.canvas.draw_idle()
                return
            
            x, y = event.xdata, event.ydata
            hover_text.set_text(f'x: {x:.4f}\ny: {y:.4f}')
            hover_text.set_visible(True)
            fig.canvas.draw_idle()
        
        # Scroll to zoom
        def on_scroll(event):
            if event.inaxes != ax:
                return
            
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            
            xdata = event.xdata
            ydata = event.ydata
            
            # Zoom factor
            if event.button == 'up':
                scale_factor = 0.8  # Zoom in
            elif event.button == 'down':
                scale_factor = 1.2  # Zoom out
            else:
                return
            
            # Calculate new limits
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
            
            ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
            ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
            
            fig.canvas.draw_idle()
        
        # Pan with mouse drag
        class PanState:
            def __init__(self):
                self.press = None
                self.xpress = None
                self.ypress = None
        
        pan_state = PanState()
        
        def on_press(event):
            if event.inaxes != ax:
                return
            pan_state.press = True
            pan_state.xpress = event.xdata
            pan_state.ypress = event.ydata
        
        def on_release(event):
            pan_state.press = None
        
        def on_pan(event):
            if not pan_state.press or event.inaxes != ax:
                return
            
            dx = event.xdata - pan_state.xpress
            dy = event.ydata - pan_state.ypress
            
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            
            new_xlim = [cur_xlim[0] - dx, cur_xlim[1] - dx]
            new_ylim = [cur_ylim[0] - dy, cur_ylim[1] - dy]
            
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
            
            fig.canvas.draw_idle()
        
        # Connect events
        fig.canvas.mpl_connect('motion_notify_event', on_move)
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_pan)
        
        # Legend
        ax.legend(fontsize=11, loc='best', facecolor='#0f3460', edgecolor='#2ECC71', labelcolor='#eee')
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=101, interval=30, blit=True, repeat=True)
        
        plt.tight_layout()
        plt.show()
        
    except ValueError as e:
        raise ValueError(f'Invalid expression for graphing. Make sure to use "x" as the variable: {str(e)}')
    except Exception as e:
        raise Exception(f'Error during graphing: {str(e)}')




def build_context():
    funcs = {
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'asinh': math.asinh,
        'acosh': math.acosh,
        'atanh': math.atanh,
        'log': math.log,
        'log10': math.log10,
        'sqrt': math.sqrt,
        'cbrt': lambda x: x ** (1/3),  # Cube root
        'exp': math.exp,
        'factorial': math.factorial,
        'abs': abs,
        'round': round,
        'pow': pow,
        'degrees': math.degrees,
        'radians': math.radians,
        'ceil': math.ceil,
        'floor': math.floor,
    }
    names = {'pi': math.pi, 'e': math.e}
    return names, funcs


def safe_eval(expr: str):
    expr = preprocess(expr)
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError('Syntax error') from e
    names, funcs = build_context()
    evaluator = SafeEvaluator(names=names, funcs=funcs)
    return evaluator.visit(tree)


def create_rounded_button_image(width, height, color, radius=15):
    """Create a rounded rectangle image for button background."""
    if Image is None:
        return None
    
    img = Image.new('RGB', (width, height), color)
    draw = ImageDraw.Draw(img)
    
    # Draw rounded rectangle
    draw.rounded_rectangle(
        [(0, 0), (width - 1, height - 1)],
        radius=radius,
        fill=color,
        outline=color
    )
    
    return img


def parse_voice_intent(text: str) -> dict:
    """Analyze voice input and determine intent (graph, scientific, normal calc, etc.)"""
    text_lower = text.lower()
    
    intent = {
        'type': 'normal',  # 'graph', 'scientific', 'normal', 'error'
        'expression': text,
        'should_graph': False,
        'should_scientific': False,
        'error': None
    }
    
    # Graph detection keywords
    graph_keywords = ['graph', 'plot', 'draw', 'chart', 'visualize', 'display graph', 'show graph', 'sketch']
    if any(keyword in text_lower for keyword in graph_keywords):
        intent['type'] = 'graph'
        intent['should_graph'] = True
        # Extract expression after graph keyword
        for keyword in graph_keywords:
            if keyword in text_lower:
                expr = text_lower.split(keyword, 1)[1].strip()
                intent['expression'] = expr if expr else text
                break
        return intent
    
    # Scientific mode detection - check for scientific functions
    scientific_keywords = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 
                          'sinh', 'cosh', 'tanh', 'log', 'ln', 'sqrt', 
                          'exp', 'factorial', 'degree', 'radian', 'pi', 'euler',
                          'squared', 'cubed', 'power', 'root', 'sine', 'cosine', 
                          'tangent', 'logarithm', 'exponential', 'absolute', 'ceiling', 'floor']
    
    if any(keyword in text_lower for keyword in scientific_keywords):
        intent['type'] = 'scientific'
        intent['should_scientific'] = True
        return intent
    
    # Check if it's a valid expression
    try:
        processed = preprocess(text)
        ast.parse(processed, mode='eval')
        intent['type'] = 'normal'
        return intent
    except:
        intent['type'] = 'error'
        intent['error'] = 'Could not parse expression'
        return intent


def handle_unified_voice_command(root, display_var, mode_var):
    """Unified voice command that determines intent and routes accordingly."""
    if sr is None:
        messagebox.showerror('Error', 'speech_recognition library not installed.\nRun: pip install SpeechRecognition')
        return
    
    recognizer = sr.Recognizer()
    
    # Create animated voice popup
    popup = tk.Toplevel(root)
    popup.title('🎤 Smart Voice Input')
    popup.geometry('500x450')
    popup.resizable(False, False)
    popup.configure(bg='#1a1a2e')
    popup.attributes('-topmost', True)
    popup.grab_set()
    
    # Main container
    main_frame = tk.Frame(popup, bg='#1a1a2e')
    main_frame.pack(fill='both', expand=True, padx=20, pady=20)
    
    # Header frame
    header_frame = tk.Frame(main_frame, bg='#0f3460', highlightbackground='#2ECC71', highlightthickness=2)
    header_frame.pack(fill='x', pady=(0, 15), ipady=15)
    
    # Status label
    status_label = tk.Label(header_frame, text='🎤 Listening...', bg='#0f3460', fg='#2ECC71',
                           font=('Segoe UI', 18, 'bold'))
    status_label.pack()
    
    # Animated dots
    dots_frame = tk.Frame(header_frame, bg='#0f3460')
    dots_frame.pack(pady=10)
    
    dots = []
    for i in range(4):
        dot = tk.Label(dots_frame, text='●', fg='#FF8C00', font=('Segoe UI', 14), bg='#0f3460')
        dot.pack(side='left', padx=3)
        dots.append(dot)
    
    anim_state = [0]
    anim_running = [True]
    
    def pulse_animation():
        if popup.winfo_exists() and anim_running[0]:
            try:
                for i, dot in enumerate(dots):
                    if (i - anim_state[0]) % 4 == 0:
                        dot.config(fg='#2ECC71')
                    else:
                        dot.config(fg='#FF8C00')
                anim_state[0] = (anim_state[0] + 1) % 4
                popup.after(150, pulse_animation)
            except:
                pass
    
    # Content frame
    content_frame = tk.Frame(main_frame, bg='#163e39', relief='flat', bd=1)
    content_frame.pack(fill='both', expand=True, pady=(0, 15))
    
    # Message label
    msg_label = tk.Label(content_frame, text='Speak your expression, calculation, or "graph"...\n\nWaiting for audio input...',
                        bg='#163e39', fg='#eee', font=('Segoe UI', 11), justify='center', wraplength=400)
    msg_label.pack(pady=25, padx=15, expand=True)
    
    # Result container (hidden initially)
    result_container = tk.Frame(content_frame, bg='#163e39')
    
    result_label = tk.Label(result_container, text='Recognized:', bg='#163e39', fg='#2ECC71',
                           font=('Segoe UI', 9, 'bold'), justify='left')
    result_label.pack(anchor='w', padx=10, pady=(5, 0))
    
    result_text = tk.Text(result_container, height=3, width=45, bg='#0f3460', fg='#2ECC71',
                         font=('Monospace', 13), relief='flat', bd=1, wrap='word', state='disabled',
                         highlightthickness=1, highlightbackground='#2ECC71')
    result_text.pack(padx=10, pady=(5, 10), fill='both', expand=True)
    
    # Button frame
    button_frame = tk.Frame(main_frame, bg='#1a1a2e')
    button_frame.pack(fill='x', pady=(0, 0))
    
    # Cancel button
    cancel_btn = tk.Button(button_frame, text='Cancel', bg='#FF8C00', fg='white',
                          font=('Segoe UI', 10, 'bold'), relief='flat', bd=0, padx=20, pady=10,
                          activebackground='#FF6B35', activeforeground='white', cursor='hand2')
    cancel_btn.pack(side='right', padx=(5, 0))
    
    def destroy_popup():
        if popup.winfo_exists():
            anim_running[0] = False
            popup.destroy()
    
    cancel_btn.config(command=destroy_popup)
    
    # Start listening animation
    pulse_animation()
    popup.update()
    
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=5)
    except sr.RequestError:
        if popup.winfo_exists():
            anim_running[0] = False
            status_label.config(text='❌ Microphone Error', fg='#FF8C00')
            msg_label.config(text='Microphone not found.\nEnsure your microphone is connected.')
            dots_frame.pack_forget()
            cancel_btn.config(text='Close')
            popup.after(3000, destroy_popup)
        return
    except sr.UnknownValueError:
        if popup.winfo_exists():
            anim_running[0] = False
            status_label.config(text='❌ No Audio Detected', fg='#FF8C00')
            msg_label.config(text='Could not detect any audio input.\nPlease try again.')
            dots_frame.pack_forget()
            cancel_btn.config(text='Retry')
            def retry_voice():
                destroy_popup()
                handle_unified_voice_command(root, display_var, mode_var)
            cancel_btn.config(command=retry_voice)
            popup.after(3000, lambda: cancel_btn.config(command=destroy_popup, text='Close', bg='#FF8C00'))
        return
    except Exception as e:
        if popup.winfo_exists():
            anim_running[0] = False
            status_label.config(text='⚠️ Error', fg='#FF8C00')
            msg_label.config(text=f'Audio error: {str(e)[:50]}')
            dots_frame.pack_forget()
            cancel_btn.config(text='Close')
            popup.after(3000, destroy_popup)
        return
    
    # Processing state
    if popup.winfo_exists():
        anim_running[0] = False
        status_label.config(text='⚙️ Processing...', fg='#2ECC71')
        msg_label.config(text='Converting speech to text...\nAnalyzing intent...')
        popup.update()
    
    try:
        # Recognize speech
        text = recognizer.recognize_google(audio)
        
        # Clean up common patterns
        text = text.lower()
        text = text.replace(' times ', ' * ').replace(' multiply ', ' * ')
        text = text.replace(' divided by ', ' / ').replace(' power ', ' ** ')
        text = text.replace(' squared', ' ** 2').replace(' cubed', ' ** 3')
        text = text.replace(' point ', '.')
        
        # Parse intent
        intent = parse_voice_intent(text)
        
        # Success state with result
        if popup.winfo_exists():
            status_label.config(text='✓ Success!', fg='#2ECC71')
            msg_label.pack_forget()
            
            # Show result
            result_container.pack(fill='both', expand=True, padx=5, pady=5)
            result_text.config(state='normal')
            result_text.delete('1.0', 'end')
            
            if intent['type'] == 'graph':
                result_text.insert('1.0', f"📈 GRAPH MODE\n\nExpression: {intent['expression']}")
            elif intent['type'] == 'scientific':
                result_text.insert('1.0', f"📊 SCIENTIFIC MODE\n\nExpression: {intent['expression']}")
            else:
                result_text.insert('1.0', f"Expression: {intent['expression']}")
            
            result_text.config(state='disabled')
            
            # Update buttons
            cancel_btn.pack_forget()
            
            confirm_frame = tk.Frame(button_frame, bg='#1a1a2e')
            confirm_frame.pack(fill='x', expand=True)
            
            def confirm_result():
                if intent['type'] == 'graph':
                    # Open graph dialog
                    destroy_popup()
                    open_graph_dialog(root, display_var)
                    # Set the expression in dialog
                    root.after(200, lambda: display_var.set(intent['expression']))
                elif intent['type'] == 'scientific':
                    # Enable scientific mode and set expression
                    destroy_popup()
                    mode_var.set(True)
                    display_var.set(intent['expression'])
                else:
                    # Normal calculation
                    try:
                        res = safe_eval(intent['expression'])
                        if isinstance(res, float):
                            res = round(res, 10)
                        display_var.set(str(res))
                    except Exception as e:
                        messagebox.showerror('Error', str(e))
                    destroy_popup()
            
            def retry_voice():
                destroy_popup()
                handle_unified_voice_command(root, display_var, mode_var)
            
            confirm_button = tk.Button(confirm_frame, text='✓ Confirm', bg='#2ECC71', fg='white',
                                      font=('Segoe UI', 11, 'bold'), relief='flat', bd=0, padx=25, pady=10,
                                      activebackground='#27AE60', cursor='hand2', command=confirm_result)
            confirm_button.pack(side='left', padx=(0, 5), fill='both', expand=True)
            
            retry_button = tk.Button(confirm_frame, text='🔄 Retry', bg='#FF8C00', fg='white',
                                    font=('Segoe UI', 11, 'bold'), relief='flat', bd=0, padx=25, pady=10,
                                    activebackground='#FF6B35', cursor='hand2', command=retry_voice)
            retry_button.pack(side='left', fill='both', expand=True)
            
            popup.update()
        
        return
        
    except sr.UnknownValueError:
        if popup.winfo_exists():
            anim_running[0] = False
            status_label.config(text='❌ Not Recognized', fg='#FF8C00')
            msg_label.config(text='Could not understand the audio.\nPlease speak clearly and try again.')
            
            def retry_click():
                destroy_popup()
                handle_unified_voice_command(root, display_var, mode_var)
            
            cancel_btn.config(text='🔄 Retry', command=retry_click, bg='#0f3460', fg='#2ECC71',
                            activebackground='#163e39')
            popup.update()
            popup.after(4000, lambda: cancel_btn.config(text='Close', command=destroy_popup, bg='#FF8C00', fg='white'))
        return
    except sr.RequestError:
        if popup.winfo_exists():
            anim_running[0] = False
            status_label.config(text='❌ Service Error', fg='#FF8C00')
            msg_label.config(text='Speech recognition service unavailable.\nCheck your internet connection.')
            cancel_btn.config(text='Close')
            popup.update()
            popup.after(3000, destroy_popup)
        return



def open_graph_dialog(root, display_var):
    """Open a dialog to create a graph of an expression."""
    if plt is None or np is None:
        messagebox.showerror('Error', 'matplotlib and numpy not installed.\nRun: pip install matplotlib numpy')
        return
    
    # Create dialog window
    dialog = tk.Toplevel(root)
    dialog.title('📈 Function Grapher')
    dialog.geometry('550x450')
    dialog.resizable(False, False)
    dialog.configure(bg='#1a1a2e')
    dialog.attributes('-topmost', True)
    dialog.grab_set()
    
    # Animation for dialog appearance
    dialog.attributes('-alpha', 0.0)
    
    def animate_appearance(alpha=0.0):
        if alpha < 1.0:
            dialog.attributes('-alpha', alpha)
            dialog.after(15, lambda: animate_appearance(alpha + 0.1))
    
    # Main frame
    main_frame = tk.Frame(dialog, bg='#1a1a2e')
    main_frame.pack(fill='both', expand=True, padx=20, pady=20)
    
    # Title
    title = tk.Label(main_frame, text='📈 Function Grapher', bg='#1a1a2e', fg='#2ECC71',
                     font=('Segoe UI', 16, 'bold'))
    title.pack(pady=(0, 15))
    
    # Expression frame
    expr_label = tk.Label(main_frame, text='Enter Expression (x = variable) or let voice decide mode:', bg='#1a1a2e', 
                         fg='#eee', font=('Segoe UI', 10))
    expr_label.pack(anchor='w', pady=(0, 5))
    
    expr_var = tk.StringVar(value=display_var.get())
    expr_entry = tk.Entry(main_frame, textvariable=expr_var, font=('Segoe UI', 12),
                         bg='#163e39', fg='#2ECC71', relief='flat', bd=1,
                         insertbackground='#FF8C00', highlightthickness=1, highlightbackground='#2ECC71')
    expr_entry.pack(fill='x', pady=(0, 15), ipady=8)
    expr_entry.focus()
    
    # Info box
    info_frame = tk.Frame(main_frame, bg='#0f3460', relief='flat', bd=1)
    info_frame.pack(fill='both', expand=True, pady=(0, 15))
    
    info_text = tk.Label(info_frame, 
                        text='Examples:\n\n• "x squared" → Graph mode\n• "graph sine of x" → Auto graph\n• "sin x times pi" → Scientific\n• "2 plus 2" → Normal calc\n\nVoice understands context!',
                        bg='#0f3460', fg='#2ECC71', font=('Segoe UI', 9), justify='left')
    info_text.pack(anchor='nw', padx=15, pady=15, expand=True)
    
    # Button frame
    btn_frame = tk.Frame(main_frame, bg='#1a1a2e')
    btn_frame.pack(fill='x')
    
    def generate_graph():
        expr = expr_var.get().strip()
        if not expr:
            messagebox.showwarning('Warning', 'Please enter an expression')
            return
        
        try:
            graph_expression(expr, title=f'Graph: {expr}')
        except Exception as e:
            messagebox.showerror('Error', str(e))
    
    def on_voice():
        """Capture voice input for expression from main calculator style popup"""
        if sr is None:
            messagebox.showerror('Error', 'speech_recognition library not installed.')
            return
        
        recognizer = sr.Recognizer()
        
        # Create voice popup similar to main calculator
        voice_popup = tk.Toplevel(dialog)
        voice_popup.title('🎤 Voice Input')
        voice_popup.geometry('400x300')
        voice_popup.resizable(False, False)
        voice_popup.configure(bg='#1a1a2e')
        voice_popup.attributes('-topmost', True)
        voice_popup.grab_set()
        
        # Main container
        voice_frame = tk.Frame(voice_popup, bg='#1a1a2e')
        voice_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Status
        status_label = tk.Label(voice_frame, text='🎤 Listening...', bg='#1a1a2e', fg='#2ECC71',
                               font=('Segoe UI', 16, 'bold'))
        status_label.pack(pady=20)
        
        # Animated dots
        dots_frame = tk.Frame(voice_frame, bg='#1a1a2e')
        dots_frame.pack(pady=10)
        
        dots = []
        for i in range(4):
            dot = tk.Label(dots_frame, text='●', fg='#FF8C00', font=('Segoe UI', 14), bg='#1a1a2e')
            dot.pack(side='left', padx=3)
            dots.append(dot)
        
        anim_state = [0]
        anim_running = [True]
        
        def pulse_animation():
            if voice_popup.winfo_exists() and anim_running[0]:
                try:
                    for i, dot in enumerate(dots):
                        if (i - anim_state[0]) % 4 == 0:
                            dot.config(fg='#2ECC71')
                        else:
                            dot.config(fg='#FF8C00')
                    anim_state[0] = (anim_state[0] + 1) % 4
                    voice_popup.after(150, pulse_animation)
                except:
                    pass
        
        # Message label
        msg_label = tk.Label(voice_frame, text='Speak your mathematical expression...\n\nWaiting for audio input...',
                            bg='#1a1a2e', fg='#eee', font=('Segoe UI', 10), justify='center')
        msg_label.pack(pady=20, expand=True)
        
        pulse_animation()
        voice_popup.update()
        
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=7, phrase_time_limit=5)
        except sr.RequestError:
            if voice_popup.winfo_exists():
                anim_running[0] = False
                status_label.config(text='❌ Microphone Error', fg='#FF8C00')
                msg_label.config(text='Microphone not found.\nEnsure your microphone is connected.')
                dots_frame.pack_forget()
                voice_popup.after(2000, lambda: voice_popup.destroy() if voice_popup.winfo_exists() else None)
            return
        except sr.UnknownValueError:
            if voice_popup.winfo_exists():
                anim_running[0] = False
                status_label.config(text='❌ No Audio Detected', fg='#FF8C00')
                msg_label.config(text='Could not detect any audio input.\nPlease try again.')
                dots_frame.pack_forget()
                voice_popup.after(2000, lambda: voice_popup.destroy() if voice_popup.winfo_exists() else None)
            return
        except Exception as e:
            if voice_popup.winfo_exists():
                anim_running[0] = False
                status_label.config(text='⚠️ Error', fg='#FF8C00')
                msg_label.config(text=f'Audio error: {str(e)[:50]}')
                dots_frame.pack_forget()
                voice_popup.after(2000, lambda: voice_popup.destroy() if voice_popup.winfo_exists() else None)
            return
        
        # Processing state
        if voice_popup.winfo_exists():
            anim_running[0] = False
            status_label.config(text='⚙️ Processing...', fg='#2ECC71')
            msg_label.config(text='Converting speech to text...\nPlease wait')
            voice_popup.update()
        
        try:
            text = recognizer.recognize_google(audio)
            
            # Clean up common voice patterns
            text = text.lower()
            text = text.replace(' times ', ' * ').replace(' multiply ', ' * ')
            text = text.replace(' divided by ', ' / ').replace(' power ', ' ** ')
            text = text.replace(' squared', ' ** 2').replace(' cubed', ' ** 3')
            text = text.replace(' point ', '.')
            
            # Success state
            if voice_popup.winfo_exists():
                status_label.config(text='✓ Success!', fg='#2ECC71')
                msg_label.config(text=f'Recognized:\n\n{text}')
                dots_frame.pack_forget()
                
                # Confirm button
                def use_expression():
                    expr_var.set(text)
                    voice_popup.destroy()
                
                confirm_btn = tk.Button(voice_frame, text='✓ Use This Expression', command=use_expression,
                                       bg='#2ECC71', fg='white', font=('Segoe UI', 10, 'bold'),
                                       relief='flat', bd=0, padx=20, pady=10, activebackground='#27AE60',
                                       cursor='hand2')
                confirm_btn.pack(pady=10)
                
                voice_popup.update()
        
        except sr.UnknownValueError:
            if voice_popup.winfo_exists():
                anim_running[0] = False
                status_label.config(text='❌ Not Recognized', fg='#FF8C00')
                msg_label.config(text='Could not understand the audio.\nPlease speak clearly and try again.')
                dots_frame.pack_forget()
                voice_popup.after(3000, lambda: voice_popup.destroy() if voice_popup.winfo_exists() else None)
        except sr.RequestError:
            if voice_popup.winfo_exists():
                anim_running[0] = False
                status_label.config(text='❌ Service Error', fg='#FF8C00')
                msg_label.config(text='Speech recognition service unavailable.\nCheck your internet connection.')
                dots_frame.pack_forget()
                voice_popup.after(3000, lambda: voice_popup.destroy() if voice_popup.winfo_exists() else None)
    
    graph_btn = tk.Button(btn_frame, text='📈 Generate Graph', command=generate_graph,
                         bg='#2ECC71', fg='white', font=('Segoe UI', 11, 'bold'),
                         relief='flat', bd=0, padx=20, pady=10, activebackground='#27AE60',
                         cursor='hand2')
    graph_btn.pack(side='left', fill='both', expand=True, padx=(0, 5))
    
    voice_btn = tk.Button(btn_frame, text='🎤 Voice', command=on_voice,
                         bg='#FF8C00', fg='white', font=('Segoe UI', 11, 'bold'),
                         relief='flat', bd=0, padx=20, pady=10, activebackground='#FF6B35',
                         cursor='hand2')
    voice_btn.pack(side='left', fill='both', expand=True, padx=(5, 0))
    
    # Animate appearance
    animate_appearance()


def run_self_test():
    tests = [
        ('2+2', 4),
        ('3^2', 9),
        ('5!', math.factorial(5)),
        ('sin(pi/2)', 1.0),
        ('ln(e)', 1.0),
        ('sqrt(16)', 4.0),
        ('x squared', None),  # Human language - requires x value
        ('x cubed', None),    # Human language - requires x value
    ]
    print('Running tests...')
    failures = 0
    for expr, expected in tests:
        try:
            if expected is None:
                # These are human language expressions that need preprocessing
                processed = preprocess(expr)
                print('OK', expr, '→', processed)
                continue
            
            got = safe_eval(expr)
            ok = abs(got - expected) < 1e-9 if isinstance(expected, float) or isinstance(got, float) else got == expected
        except Exception as e:
            print('FAILED', expr, 'error', e)
            failures += 1
            continue
        if not ok:
            print('FAILED', expr, 'expected', expected, 'got', got)
            failures += 1
        else:
            print('OK', expr, '=', got)
    if failures:
        print(f'{failures} failed')
        sys.exit(2)
    print('All tests passed')


def launch_gui():
    if tk is None:
        print('Tkinter not available in this Python. Install or use the web UI.')
        return
    root = tk.Tk()
    root.title('Scientific Calculator')
    root.geometry('500x750')
    root.resizable(False, False)
    
    # Dark theme with orange and green colors
    BG_COLOR = '#1a1a2e'
    FG_COLOR = '#eee'
    NAVY_BLUE = '#0f3460'
    ACCENT_BLUE = "#163e39"
    BUTTON_BLUE = '#0f3460'
    HIGHLIGHT = '#FF8C00'
    
    root.configure(bg=BG_COLOR)

    frm = tk.Frame(root, bg=BG_COLOR, padx=10, pady=10)
    frm.pack(fill='both', expand=True)
    
    # Top control panel
    control_frm = tk.Frame(frm, bg=BG_COLOR)
    control_frm.pack(fill='x', pady=(0, 10))
    
    # Title
    title_lbl = tk.Label(control_frm, text='Calculator', bg=BG_COLOR, fg='#2ECC71', font=('Segoe UI', 16, 'bold'))
    title_lbl.pack(side='left')
    
    # Mode toggle
    mode_var = tk.BooleanVar(value=False)
    def toggle_mode():
        mode_var.set(not mode_var.get())
        recreate_buttons()
        update_title()
    
    def update_title():
        mode_text = 'Scientific' if mode_var.get() else 'Normal'
        title_lbl.config(text=f'{mode_text} Calculator')
    
    mode_btn = tk.Button(control_frm, text='📊 Scientific', command=toggle_mode,
                        bg=NAVY_BLUE, fg='#2ECC71', font=('Segoe UI', 9, 'bold'),
                        relief='flat', bd=0, padx=10, pady=5, activebackground='#FF8C00',
                        activeforeground=BG_COLOR, cursor='hand2')
    mode_btn.pack(side='right', padx=(10, 0))
    
    # Voice command button
    def on_voice_command():
        handle_unified_voice_command(root, display_var, mode_var)
    
    voice_btn = tk.Button(control_frm, text='🎤 Smart Voice', command=on_voice_command,
                         bg=NAVY_BLUE, fg='#2ECC71', font=('Segoe UI', 9, 'bold'),
                         relief='flat', bd=0, padx=10, pady=5, activebackground='#FF8C00',
                         activeforeground=BG_COLOR, cursor='hand2')
    voice_btn.pack(side='right', padx=(5, 0))
    
    # Graph button
    def on_graph():
        open_graph_dialog(root, display_var)
    
    graph_btn = tk.Button(control_frm, text='📈 Graph', command=on_graph,
                         bg=NAVY_BLUE, fg='#2ECC71', font=('Segoe UI', 9, 'bold'),
                         relief='flat', bd=0, padx=10, pady=5, activebackground='#FF8C00',
                         activeforeground=BG_COLOR, cursor='hand2')
    graph_btn.pack(side='right', padx=(5, 0))

    display_var = tk.StringVar()
    entry = tk.Entry(frm, textvariable=display_var, font=('Segoe UI', 26), justify='right',
                     bg=ACCENT_BLUE, fg=FG_COLOR, bd=0, relief='flat', insertbackground=HIGHLIGHT)
    entry.pack(fill='x', pady=(0, 10), ipady=12)

    history = tk.Listbox(frm, height=4, bg=ACCENT_BLUE, fg=FG_COLOR, font=('Segoe UI', 9),
                         bd=0, relief='flat', selectmode='none')
    history.pack(fill='both', expand=True, pady=(0, 10))
    
    # History label
    hist_lbl = tk.Label(frm, text='History', bg=BG_COLOR, fg='#2ECC71', font=('Segoe UI', 10, 'bold'))
    hist_lbl.pack(anchor='w', pady=(0, 2))

    btn_frame = tk.Frame(frm, bg=BG_COLOR)
    btn_frame.pack(fill='both', expand=True)
    
    # Button definitions
    normal_buttons = [
        ('C', 'clear', '#2ECC71'), ('DEL','del', '#2ECC71'), ('(', '(', BUTTON_BLUE), (')',')' , BUTTON_BLUE),
        ('7','7', BUTTON_BLUE),('8','8', BUTTON_BLUE),('9','9', BUTTON_BLUE),('/','/','#FF8C00'),
        ('4','4', BUTTON_BLUE),('5','5', BUTTON_BLUE),('6','6', BUTTON_BLUE),('*','*','#FF8C00'),
        ('1','1', BUTTON_BLUE),('2','2', BUTTON_BLUE),('3','3', BUTTON_BLUE),('-','-','#FF8C00'),
        ('0','0', BUTTON_BLUE),('.','.',BUTTON_BLUE),('^','^','#FF8C00'),('+','+','#FF8C00'),
    ]

    scientific_buttons = [
        ('C', 'clear', '#2ECC71'), ('DEL','del', '#2ECC71'), ('(', '(', BUTTON_BLUE), (')',')' , BUTTON_BLUE),
        ('7','7', BUTTON_BLUE),('8','8', BUTTON_BLUE),('9','9', BUTTON_BLUE),('/','/','#FF8C00'),
        ('4','4', BUTTON_BLUE),('5','5', BUTTON_BLUE),('6','6', BUTTON_BLUE),('*','*','#FF8C00'),
        ('1','1', BUTTON_BLUE),('2','2', BUTTON_BLUE),('3','3', BUTTON_BLUE),('-','-','#FF8C00'),
        ('0','0', BUTTON_BLUE),('.','.',BUTTON_BLUE),('^','^','#FF8C00'),('+','+','#FF8C00'),
        ('π','pi', NAVY_BLUE),('e','e', NAVY_BLUE),('n!','!', NAVY_BLUE),('=','eval','#2ECC71'),
        ('sin','sin(', NAVY_BLUE),('cos','cos(', NAVY_BLUE),('tan','tan(', NAVY_BLUE),('√','sqrt(', NAVY_BLUE),
        ('ln','ln(', NAVY_BLUE),('log₁₀','log10(', NAVY_BLUE),('sinh','sinh(', NAVY_BLUE),('cosh','cosh(', NAVY_BLUE),
        ('sinh⁻¹','asinh(', NAVY_BLUE),('cosh⁻¹','acosh(', NAVY_BLUE),('tanh⁻¹','atanh(', NAVY_BLUE),('%','%', '#FF8C00'),
        ('deg→rad','deg_to_rad(', NAVY_BLUE),('rad→deg','rad_to_deg(', NAVY_BLUE),('deg','degrees(', NAVY_BLUE),('rad','radians(', NAVY_BLUE),
    ]

    def on_button(v):
        current = display_var.get()
        if v == 'clear':
            display_var.set('')
            return
        if v == 'del':
            display_var.set(current[:-1])
            return
        if v == 'eval':
            expr = current
            try:
                res = safe_eval(expr)
                if isinstance(res, float):
                    res = round(res, 10)
                history.insert(0, f'{expr} = {res}')
                display_var.set(str(res))
            except Exception as e:
                messagebox.showerror('Error', str(e))
            return
        if v == 'pi':
            display_var.set(current + str(math.pi))
            return
        if v == 'e':
            display_var.set(current + str(math.e))
            return
        if v == '!':
            display_var.set(current + '!')
            return
        if v == 'deg_to_rad(':
            display_var.set(current + 'radians(')
            return
        if v == 'rad_to_deg(':
            display_var.set(current + 'degrees(')
            return
        display_var.set(current + v)

    def recreate_buttons():
        # Clear old buttons
        for widget in btn_frame.winfo_children():
            widget.destroy()
        
        btn_frame.columnconfigure(0, weight=0)
        btn_frame.rowconfigure(0, weight=0)
        
        # Determine which buttons to show
        buttons = scientific_buttons if mode_var.get() else normal_buttons
        
        # Also need to add equals button for normal mode
        if not mode_var.get():
            buttons.append(('=','eval','#2ECC71'))
        
        # Create grid of buttons with modern styling and rounded corners
        r = 0; c = 0
        cols = 4
        for text, val, color in buttons:
            # Create button with rounded corners effect
            b = tk.Button(btn_frame, text=text, command=lambda v=val: on_button(v),
                         bg=color, fg=FG_COLOR, font=('Segoe UI', 11, 'bold'),
                         relief='flat', bd=0, activebackground=HIGHLIGHT, activeforeground=BG_COLOR,
                         padx=0, pady=0, cursor='hand2', highlightthickness=0,
                         overrelief='flat')
            
            b.grid(row=r, column=c, ipadx=12, ipady=14, padx=2, pady=2, sticky='nsew')
            
            # Bind hover effect for better visual feedback
            def on_enter(event, btn=b, orig_color=color):
                if event.widget == btn:
                    btn.config(bg='#FF6B35')
            def on_leave(event, btn=b, orig_color=color):
                if event.widget == btn:
                    btn.config(bg=orig_color)
            
            b.bind('<Enter>', on_enter)
            b.bind('<Leave>', on_leave)
            
            c += 1
            if c == cols:
                c = 0; r += 1

        for i in range(cols): 
            btn_frame.columnconfigure(i, weight=1)
        for i in range(r+1): 
            btn_frame.rowconfigure(i, weight=1)

    # Initial button creation
    recreate_buttons()

    def on_enter(e):
        try:
            expr = display_var.get()
            res = safe_eval(expr)
            if isinstance(res, float):
                res = round(res, 10)
            history.insert(0, f'{expr} = {res}')
            display_var.set(str(res))
        except Exception as ex:
            messagebox.showerror('Error', str(ex))

    entry.bind('<Return>', on_enter)
    root.mainloop()


if __name__ == '__main__':
    if '--test' in sys.argv:
        run_self_test()
    else:
        launch_gui()
