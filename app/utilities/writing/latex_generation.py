import subprocess
import os
from tempfile import NamedTemporaryFile

latex_template = r"""
    \documentclass{{article}}
    \usepackage{{amsmath}}
    \usepackage{{slashed}}    
    \usepackage{{hyperref}}
    \usepackage{{tensor}}
    \usepackage{{colortbl}}
    \usepackage{{xcolor}}
    \usepackage{{booktabs}}

    \definecolor{{rowgray}}{{gray}}{{0.9}}
    \rowcolor{{1}}{{white}}{{rowgray}}

    
    \begin{{document}}

    \section*{{Introduction}}
    
    The \textbf{{{experiment_name}}} seeks to probe the \textbf{{Underlying Equation}} that is conjectured to represent a fancy 
    bullshit relationship between $x$ and $y$:
    \begin{{equation}}
        \boxed{{{underlying_equation_content}}}.
    \end{{equation}}

    \section*{{Generated Table with Alternating Colors}}

    We have obtained the following raw data:
    {experimental_data_table}
    \end{{document}}
"""

def generate_document(
        underlying_equation: str,
        experimental_data_table: str,
        experiment_name: str = 'E0001 Experiment'):

    latex_code = latex_template.format(
        experiment_name = experiment_name,
        underlying_equation_content = underlying_equation,
        experimental_data_table = experimental_data_table)

    with NamedTemporaryFile(delete = False, suffix = ".tex") as tex_file:
        tex_filename = tex_file.name
        tex_file.write(latex_code.encode("utf-8"))

    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_filename], check = True)
        os.rename(tex_filename.replace(".tex", ".pdf"), "output.pdf")
    except subprocess.CalledProcessError as E:
        print(E)
    finally:
        for extension in [".aux", ".log", ".tex"]:
            try:
                os.remove(tex_filename.replace(".tex", extension))
            except FileNotFoundError:
                pass