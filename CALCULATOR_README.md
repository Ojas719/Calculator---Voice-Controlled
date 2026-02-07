# Scientific Calculator with Voice Commands

A modern Tkinter-based scientific calculator with voice input capabilities and rounded button design.

## Features

- **Voice Commands**: Speak your mathematical expression and have it automatically recognized and solved
- **Scientific Mode**: Toggle between normal and advanced scientific functions
- **Calculation History**: View your recent calculations
- **Modern UI**: Dark theme with smooth visual design

## Installation

1. Install required dependencies by running:
   ```bash
   python "c:\Users\OJAS RANA\OneDrive\Documents\setup_calculator.py"
   ```

   Or manually:
   ```bash
   pip install SpeechRecognition pillow
   ```

2. Ensure you have a working microphone (required for voice commands)

## Usage

### GUI Mode (Interactive)
```bash
python "c:\Users\OJAS RANA\OneDrive\Documents\calculator_tk.py"
```

### Command Line Tests
```bash
python "c:\Users\OJAS RANA\OneDrive\Documents\calculator_tk.py" --test
```

## Voice Commands

Click the **🎤 Voice** button and speak your expression naturally:

- "two plus two" → 2+2 = 4
- "five times three" → 5*3 = 15
- "ten divided by two" → 10/2 = 5
- "five squared" → 5**2 = 25
- "three cubed" → 3**3 = 27
- "sine of pi divided by two" → sin(pi/2) = 1.0
- "square root of sixteen" → sqrt(16) = 4.0

The voice system automatically converts common phrases:
- "times" / "multiply" → *
- "divided by" / "divide" → /
- "plus" → +
- "minus" → -
- "to the power of" → **
- "squared" → ** 2
- "cubed" → ** 3

## Button Design

All buttons feature:
- Smooth flat design with hover effects
- Rounded appearance
- Color-coded operations (blue for numbers, red/cyan for operations)
- Responsive click feedback

## Supported Functions

### Basic Operations
- Addition (+), Subtraction (-), Multiplication (*), Division (/)
- Power (^)
- Modulo (%)
- Factorial (!)

### Trigonometric Functions
- sin, cos, tan
- asin, acos, atan
- sinh, cosh, tanh
- asinh, acosh, atanh

### Logarithmic Functions
- log (natural logarithm)
- log10 (base-10 logarithm)

### Other Functions
- sqrt (square root)
- exp (exponential)
- abs (absolute value)
- round, ceil, floor
- degrees, radians
- Constants: π (pi), e

## Requirements

- Python 3.6+
- Tkinter (usually included with Python)
- SpeechRecognition (for voice input)
- Pillow (for enhanced button styling)

## Troubleshooting

**Microphone not found:**
- Check that your microphone is properly connected
- Verify microphone permissions in your system settings

**"Could not understand audio":**
- Speak more clearly
- Reduce background noise
- Ensure the microphone is close enough

**Google Speech Recognition unavailable:**
- Check your internet connection (voice recognition uses Google's API)
- Try again after a moment

## License

Use freely for personal and commercial purposes.
