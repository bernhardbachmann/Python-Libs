
import sympy as sp
import numpy as np
import mpmath
import matplotlib.pyplot as plt

class FolgenBibliothek:
    @staticmethod
    def print_symbolic_sequence_table(sequence_expr, n_symbol, n_start=1, n_end=10):
        header_sym = "a_n (Symbolisch)"
        symbolic_values = [str(sequence_expr.subs(n_symbol, k)) for k in range(n_start, n_end + 1)]
        col_width_sym = max(len(header_sym), max(len(s) for s in symbolic_values))

        print(f"Symbolische Tabelle der Folge a_n = {sp.pretty(sequence_expr)}")
        print("-" * (col_width_sym + 10))
        print(f"{'n':<5} | {header_sym:<{col_width_sym}}")
        print("-" * (col_width_sym + 10))

        for k in range(n_start, n_end + 1):
            term_symbolic = sequence_expr.subs(n_symbol, k)
            print(f"{k:<5} | {str(term_symbolic):<{col_width_sym}}")

        print("-" * (col_width_sym + 10))

    @staticmethod
    def print_numeric_sequence_table(sequence_expr, n_symbol, n_start=1, n_end=10, precision=30):
        mpmath.mp.dps = precision + 5

        header_num = f"a_n (Numerisch, {precision} Stellen)"

        print(f"Numerische Tabelle der Folge a_n = {sp.pretty(sequence_expr)}")
        print("-" * (len(header_num) + 10))
        print(f"{'n':<5} | {header_num}")
        print("-" * (len(header_num) + 10))

        for k in range(n_start, n_end + 1):
            term_numeric = sequence_expr.subs(n_symbol, k).evalf(precision)
            numeric_str = f"{term_numeric:.{precision}f}"
            print(f"{k:<5} | {numeric_str}")

        print("-" * (len(header_num) + 10))

    @staticmethod
    def get_numeric_sequence_values(sequence_expr, n_symbol, n_start, n_end):
        n_values = list(range(n_start, n_end + 1))
        y_values = np.array([float(sequence_expr.subs(n_symbol, k).evalf()) for k in n_values])
        return n_values, y_values

    @staticmethod
    def plot_xy(sequence_expr, n_symbol, n_start=1, n_end=20):
        _, y_values = FolgenBibliothek.get_numeric_sequence_values(sequence_expr, n_symbol, n_start, n_end)
        x0 = [0.0] * len(y_values)

        plt.figure(figsize=(8, 4))
        plt.axhline(0, color='black', linewidth=1.0)
        plt.plot(y_values, x0, "o", label=f"Werte von a_n für n={n_start}..{n_end}")
        plt.xlabel("Wert a_n")
        plt.yticks([])  # y-Achse ausblenden, da sie nur 0 enthält
        plt.title("Folge als Punktwolke auf der x-Achse")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_indexwert(sequence_expr, n_symbol, n_start=1, n_end=20):
        n_values, y_values = FolgenBibliothek.get_numeric_sequence_values(sequence_expr, n_symbol, n_start, n_end)

        plt.figure(figsize=(8, 4))
        plt.plot(n_values, y_values, "o-", label=f"$a_n = {sp.latex(sequence_expr)}$")
        plt.xlabel("Index n")
        plt.ylabel("Wert a_n")
        plt.title(f"Folge a_n gegen Index n (für n={n_start}..{n_end})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_convergence_auto(sequence_expr, n_symbol, n_start=1, n_end=50, epsilon=0.1):
        n_values, a_vals = FolgenBibliothek.get_numeric_sequence_values(sequence_expr, n_symbol, n_start, n_end)

        L_symbolic = None
        try:
            L_symbolic = sp.limit(sequence_expr, n_symbol, sp.oo)
            L = float(L_symbolic.evalf())
            limit_label = fr'Grenzwert L = ${sp.latex(L_symbolic)}$'
        except (NotImplementedError, ValueError, TypeError):
            L = a_vals[-1]
            limit_label = f'Grenzwert L  {L:.4f} (numerisch)'
            print(f"Warnung: Symbolischer Grenzwert fr '{sequence_expr}' konnte nicht berechnet werden. Nutze numerische Nherung.")

        plt.figure(figsize=(10, 5))
        plt.plot(n_values, a_vals, 'bo-', label=fr'Folge $a_n = {sp.latex(sequence_expr)}$')
        plt.axhline(L, color='black', linewidth=1.5, label=limit_label)
        plt.axhline(L + epsilon, color='red', linestyle='--', label=fr'$d$-Band ($d={epsilon}$)')
        plt.axhline(L - epsilon, color='red', linestyle='--')

        N_idx = -1
        for i in range(len(a_vals)):
            if np.all(np.abs(a_vals[i:] - L) < epsilon):
                N_idx = i
                break

        if N_idx != -1:
            n_start_conv = n_values[N_idx]
            plt.scatter(n_values[N_idx:], a_vals[N_idx:], color='green', zorder=5, label=fr'Alle $|a_n - L| < d$ fr $n  {n_start_conv}$')

        plt.xlabel('Index $n$')
        plt.ylabel('Wert $a_n$')
        plt.title(r'Visualisierung des $d$-Konvergenzkriteriums')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_multiple_sequences(sequences_to_plot, n_symbol, n_start=1, n_end=50, title="Vergleich mehrerer Folgen"):
        plt.figure(figsize=(12, 7))

        for seq_expr, label in sequences_to_plot:
            n_values, y_values = FolgenBibliothek.get_numeric_sequence_values(seq_expr, n_symbol, n_start, n_end)
            plt.plot(n_values, y_values, 'o-', markersize=4, label=label)

        plt.xlabel('Index $n$')
        plt.ylabel('Wert $a_n$')
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
