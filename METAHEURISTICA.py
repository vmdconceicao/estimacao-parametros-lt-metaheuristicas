"""
Trabalho final - Metaheuristicas (PPGEE/UFMT, 1s/2026)
Estimacao dos parametros R', X', C', G' de uma linha de transmissao a partir
de medicoes fasoriais ruidosas, tratada como um problema de otimizacao e
resolvida por quatro algoritmos: Quasi-Newton, Simulated Annealing,
Differential Evolution e PSO.

Autor: Vinicius Marcos Domingues Conceicao

Uso: deixar este arquivo na mesma pasta que 'dados_SNR30.csv' e executar
     python METAHEURISTICA.py
     python METAHEURISTICA.py --rapido   (versao curta, so para testar)

Pacotes: numpy, pandas, scipy, matplotlib, scikit-posthocs
"""

import sys
import io
import time
from pathlib import Path

# O terminal do Windows costuma usar cp1252 e quebra ao imprimir simbolos
# como theta ou setas. Reconfigura a saida para utf-8 e, se ainda assim
# falhar, troca os caracteres problematicos por equivalentes simples.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                      errors="replace")
    except Exception:
        pass

_TROCAS = {
    "θ": "theta", "ℓ": "l", "²": "2", "³": "3", "†": "+", "≈": "~",
    "á": "a", "â": "a", "ã": "a", "à": "a", "é": "e", "ê": "e", "í": "i",
    "ó": "o", "ô": "o", "õ": "o", "ú": "u", "ç": "c",
    "Á": "A", "Â": "A", "Ã": "A", "É": "E", "Í": "I", "Ó": "O", "Ç": "C",
}


def imprimir(*args, **kwargs):
    texto = " ".join(str(a) for a in args)
    try:
        print(texto, **kwargs)
    except UnicodeEncodeError:
        for a, b in _TROCAS.items():
            texto = texto.replace(a, b)
        print(texto.encode("ascii", "replace").decode("ascii"), **kwargs)


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import (friedmanchisquare, wilcoxon, rankdata,
                         studentized_range)

try:
    import scikit_posthocs as sp
    TEM_POSTHOCS = True
except ImportError:
    TEM_POSTHOCS = False

PASTA = Path(__file__).resolve().parent

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10, "figure.dpi": 110})


# ------------------------------------------------------------------------------
# Configuracoes gerais
# ------------------------------------------------------------------------------
MODO_RAPIDO = "--rapido" in sys.argv

FREQ_HZ = 60.0
OMEGA = 2.0 * np.pi * FREQ_HZ
COMPRIMENTO_KM = 200.0

# Parametros verdadeiros da linha (usados como referencia para o MAPE)
THETA_REAL = np.array([0.050, 0.488, 8.94e-9, 0.0])      # R', X', C', G'
NOMES_PARAM = ["R'", "X'", "C'", "G'"]

ALGORITMOS = ["Q-Newton", "SA", "DE", "PSO"]
COR = {"Q-Newton": "#DC2626", "SA": "#F59E0B",
       "DE": "#10B981", "PSO": "#2563EB"}

CENARIOS = {"facil": (0.95, 1.05), "medio": (0.90, 1.10),
            "dificil": (0.80, 1.20)}
NOME_CENARIO = {"facil": "FÁCIL", "medio": "MÉDIO", "dificil": "DIFÍCIL"}
LISTA_CENARIOS = ["facil", "medio", "dificil"]

N_REPETICOES = 5 if MODO_RAPIDO else 30
MAX_AVALIACOES = 3000
ALPHA = 0.05


# ------------------------------------------------------------------------------
# 1. Modelo fisico da linha e funcao objetivo
# ------------------------------------------------------------------------------
def matriz_abcd(theta):
    # Calcula os quatro elementos da matriz ABCD para um dado theta.
    resist, reat, capac, condut = theta
    z = resist + 1j * reat
    y = condut + 1j * OMEGA * capac
    if z == 0 or y == 0:
        return 1 + 0j, 0 + 0j, 0 + 0j, 1 + 0j
    gama_l = np.sqrt(z * y) * COMPRIMENTO_KM
    zc = np.sqrt(z / y)
    cosh, senh = np.cosh(gama_l), np.sinh(gama_l)
    return cosh, -zc * senh, -(1 / zc) * senh, cosh


def prever_VL_IL(theta, v0, i0):
    a, b, c, d = matriz_abcd(theta)
    return a * v0 + b * i0, c * v0 + d * i0


def carregar_dados(caminho):
    # O CSV usa ';' como separador e guarda os fasores como texto complexo.
    df = pd.read_csv(caminho, sep=";")
    for col in ["V0", "I0", "VL", "IL"]:
        df[col] = df[col].apply(lambda s: complex(str(s).replace(" ", "")))
    return {c: df[c].to_numpy(dtype=complex) for c in ["V0", "I0", "VL", "IL"]}


def construir_objetivo(dados):
    # Devolve a funcao f(theta): erro quadratico medio normalizado entre
    # os fasores medidos e os previstos pelo modelo.
    v0, i0, vl, il = dados["V0"], dados["I0"], dados["VL"], dados["IL"]
    vref2 = np.mean(np.abs(vl) ** 2)
    iref2 = np.mean(np.abs(il) ** 2)

    def f(theta):
        vl_prev, il_prev = prever_VL_IL(theta, v0, i0)
        if not (np.all(np.isfinite(vl_prev)) and np.all(np.isfinite(il_prev))):
            return 1e12
        return (np.mean(np.abs(vl - vl_prev) ** 2) / vref2 +
                np.mean(np.abs(il - il_prev) ** 2) / iref2)
    return f


def mape(theta_estimado):
    # Erro percentual de R', X', C'. G' tem valor real zero, entao nao se
    # calcula percentual (usa-se o erro absoluto |G'| em outro lugar).
    erro = np.full(4, np.nan)
    for i in range(3):
        erro[i] = abs((theta_estimado[i] - THETA_REAL[i]) / THETA_REAL[i]) * 100
    return erro


def caixa_de_busca(theta_ref, cenario):
    # Limites inferior e superior de cada parametro para o cenario dado.
    fator_lo, fator_hi = CENARIOS[cenario]
    lo = fator_lo * theta_ref.copy()
    hi = fator_hi * theta_ref.copy()
    if theta_ref[3] == 0:                       # G' centrado em zero
        amplitude = 0.20 * OMEGA * theta_ref[2]
        lo[3], hi[3] = -amplitude, amplitude
    return lo, hi


# ------------------------------------------------------------------------------
# 2. Suporte aos otimizadores: controle de orcamento e historico
# ------------------------------------------------------------------------------
class OrcamentoExcedido(Exception):
    pass


class Avaliador:
    # Envolve a funcao objetivo para contar avaliacoes, guardar o melhor
    # resultado ate o momento e interromper quando o orcamento acaba.
    def __init__(self, f, max_aval, intervalo=10):
        self.f = f
        self.max_aval = max_aval
        self.intervalo = intervalo
        self.n_aval = 0
        self.melhor = np.inf
        self.melhor_theta = None
        self.historico = []

    def __call__(self, theta):
        if self.n_aval >= self.max_aval:
            raise OrcamentoExcedido
        valor = self.f(np.asarray(theta, dtype=float))
        self.n_aval += 1
        if valor < self.melhor:
            self.melhor = valor
            self.melhor_theta = np.asarray(theta, dtype=float).copy()
        if self.n_aval % self.intervalo == 0 or self.n_aval == 1:
            self.historico.append((self.n_aval, self.melhor))
        return valor


def _resultado(av, algoritmo):
    return {"theta": av.melhor_theta, "f": av.melhor,
            "historico": np.array(av.historico) if av.historico
                         else np.empty((0, 2)),
            "n_aval": av.n_aval, "algoritmo": algoritmo}


# Quasi-Newton: L-BFGS-B com reinicios aleatorios ate esgotar o orcamento.
def quasi_newton(f, lo, hi, max_aval, semente):
    rng = np.random.default_rng(semente)
    av = Avaliador(f, max_aval)
    limites = list(zip(lo, hi))
    try:
        minimize(av, rng.uniform(lo, hi), method="L-BFGS-B",
                 bounds=limites, options={"maxiter": 200, "ftol": 1e-12})
        while av.n_aval < max_aval:
            minimize(av, rng.uniform(lo, hi), method="L-BFGS-B",
                     bounds=limites, options={"maxiter": 200, "ftol": 1e-12})
    except OrcamentoExcedido:
        pass
    return _resultado(av, "Q-Newton")


# Simulated Annealing: vizinhanca gaussiana e resfriamento geometrico.
def simulated_annealing(f, lo, hi, max_aval, semente,
                        alpha=0.995, passos_por_temp=20, sigma_frac=0.10):
    rng = np.random.default_rng(semente)
    av = Avaliador(f, max_aval)
    n = len(lo)
    sigma = sigma_frac * (hi - lo)
    theta = rng.uniform(lo, hi)
    try:
        f_atual = av(theta)
        # Calibra a temperatura inicial para aceitar ~80% das pioras.
        deltas, t, ft = [], theta.copy(), f_atual
        for _ in range(50):
            tn = np.clip(t + rng.normal(0, sigma, n), lo, hi)
            fn = av(tn)
            if fn > ft:
                deltas.append(fn - ft)
            t, ft = tn, fn
        temp = (np.mean(deltas) / np.log(1 / 0.8)) if deltas else 1e-3
        theta, f_atual = t, ft
        while av.n_aval < max_aval:
            for _ in range(passos_por_temp):
                if av.n_aval >= max_aval:
                    break
                tn = np.clip(theta + rng.normal(0, sigma, n), lo, hi)
                fn = av(tn)
                delta = fn - f_atual
                if delta < 0 or rng.random() < np.exp(-delta / (temp + 1e-30)):
                    theta, f_atual = tn, fn
            temp *= alpha
            if temp < 1e-12:
                break
    except OrcamentoExcedido:
        pass
    return _resultado(av, "SA")


# Differential Evolution, esquema rand/1/bin.
def differential_evolution(f, lo, hi, max_aval, semente, NP=30, F=0.7, CR=0.9):
    rng = np.random.default_rng(semente)
    av = Avaliador(f, max_aval)
    n = len(lo)
    populacao = rng.uniform(lo, hi, size=(NP, n))
    try:
        aptidao = np.array([av(ind) for ind in populacao])
        while av.n_aval < max_aval:
            for i in range(NP):
                if av.n_aval >= max_aval:
                    break
                outros = [j for j in range(NP) if j != i]
                r1, r2, r3 = rng.choice(outros, 3, replace=False)
                doador = np.clip(populacao[r1] +
                                 F * (populacao[r2] - populacao[r3]), lo, hi)
                j_fixo = rng.integers(n)
                troca = rng.random(n) < CR
                troca[j_fixo] = True
                tentativa = np.where(troca, doador, populacao[i])
                f_tent = av(tentativa)
                if f_tent < aptidao[i]:
                    populacao[i], aptidao[i] = tentativa, f_tent
    except OrcamentoExcedido:
        pass
    return _resultado(av, "DE")


# PSO com fator de constricao de Clerc.
def particle_swarm(f, lo, hi, max_aval, semente, NP=30, phi1=2.05, phi2=2.05):
    rng = np.random.default_rng(semente)
    av = Avaliador(f, max_aval)
    n = len(lo)
    phi = phi1 + phi2
    chi = 2.0 / abs(2 - phi - np.sqrt(phi * phi - 4 * phi))   # ~0.7298
    vmax = 0.5 * (hi - lo)
    pos = rng.uniform(lo, hi, size=(NP, n))
    vel = rng.uniform(-vmax, vmax, size=(NP, n))
    try:
        ap = np.array([av(p) for p in pos])
        melhor_pessoal, f_pessoal = pos.copy(), ap.copy()
        ig = int(np.argmin(f_pessoal))
        melhor_global, f_global = melhor_pessoal[ig].copy(), f_pessoal[ig]
        while av.n_aval < max_aval:
            r1 = rng.random((NP, n))
            r2 = rng.random((NP, n))
            vel = chi * (vel + phi1 * r1 * (melhor_pessoal - pos) +
                         phi2 * r2 * (melhor_global - pos))
            vel = np.clip(vel, -vmax, vmax)
            pos = np.clip(pos + vel, lo, hi)
            for i in range(NP):
                if av.n_aval >= max_aval:
                    break
                fi = av(pos[i])
                if fi < f_pessoal[i]:
                    melhor_pessoal[i], f_pessoal[i] = pos[i].copy(), fi
                    if fi < f_global:
                        melhor_global, f_global = pos[i].copy(), fi
    except OrcamentoExcedido:
        pass
    return _resultado(av, "PSO")


OTIMIZADORES = {"Q-Newton": quasi_newton, "SA": simulated_annealing,
                "DE": differential_evolution, "PSO": particle_swarm}


# ------------------------------------------------------------------------------
# 3. Analise de paisagem (FDC, autocorrelacao e cortes 2D)
# ------------------------------------------------------------------------------
def analise_paisagem(f, cenario="dificil"):
    imprimir("\n" + "=" * 72)
    imprimir(f"  [2/4]  Analise de paisagem - cenario {NOME_CENARIO[cenario]}")
    imprimir("=" * 72)
    rng = np.random.default_rng(42)
    lo, hi = caixa_de_busca(THETA_REAL, cenario)
    n_amostras = 1000 if MODO_RAPIDO else 5000
    n_passos = 500 if MODO_RAPIDO else 2000

    # Amostragem uniforme da caixa
    amostras = rng.uniform(lo, hi, size=(n_amostras, 4))
    valores = np.array([f(t) for t in amostras])
    f_real = f(THETA_REAL)

    # FDC: correlacao de Pearson entre f e a distancia ao otimo
    largura = hi - lo
    dist = np.linalg.norm((amostras - THETA_REAL) / largura, axis=1)
    df_, dd = valores - valores.mean(), dist - dist.mean()
    fdc = (df_ * dd).sum() / (np.sqrt((df_ ** 2).sum() * (dd ** 2).sum()) + 1e-30)

    # Autocorrelacao ao longo de um passeio aleatorio
    theta = (lo + hi) / 2
    serie = np.empty(n_passos + 1)
    serie[0] = f(theta)
    for t in range(1, n_passos + 1):
        theta = np.clip(theta + rng.normal(0, 0.02 * largura), lo, hi)
        serie[t] = f(theta)
    centrada = serie - serie.mean()
    variancia = (centrada ** 2).mean()
    atrasos = np.arange(0, 80)
    autocorr = np.array([1.0 if k == 0
                         else (centrada[:-k] * centrada[k:]).mean() / (variancia + 1e-30)
                         for k in atrasos])
    comp_corr = -1.0 / np.log(autocorr[1]) if autocorr[1] > 0 else 0.0

    imprimir(f"    f(theta*) = {f_real:.4e}   FDC = {fdc:+.3f}   "
             f"comp. correlacao = {comp_corr:.1f} passos")
    imprimir(f"    razao max/min de f = {valores.max() / valores.min():.0f}")

    # Cortes 2D da superficie
    pares = [(0, 1, "R'", "X'"), (0, 2, "R'", "C'"), (1, 2, "X'", "C'")]
    res = 40 if MODO_RAPIDO else 50
    cortes = []
    for i, j, ni, nj in pares:
        eixo_i = np.linspace(lo[i], hi[i], res)
        eixo_j = np.linspace(lo[j], hi[j], res)
        gi, gj = np.meshgrid(eixo_i, eixo_j)
        z = np.empty_like(gi)
        for a in range(res):
            for b in range(res):
                t = THETA_REAL.copy()
                t[i], t[j] = gi[a, b], gj[a, b]
                z[a, b] = f(t)
        cortes.append((gi, gj, z, ni, nj, i, j))

    # Figura com seis paineis
    fig, eixos = plt.subplots(2, 3, figsize=(13, 7.5))
    ax = eixos[0, 0]
    ax.scatter(dist, valores, s=4, alpha=0.3, color="#2563EB")
    ax.set_xlabel(r"$d(\theta,\theta^*)$  (normalizada)")
    ax.set_ylabel(r"$f(\theta)$")
    ax.set_yscale("log")
    ax.set_title(f"FDC = {fdc:+.3f}")
    ax.grid(alpha=0.3)

    ax = eixos[0, 1]
    ax.hist(np.log10(valores), bins=40, color="#10B981", edgecolor="white")
    ax.axvline(np.log10(f_real), color="red", linestyle="--",
               label=fr"$f(\theta^*)={f_real:.2e}$")
    ax.set_xlabel(r"$\log_{10} f(\theta)$")
    ax.set_ylabel("frequência")
    ax.set_title(f"Distribuição de $f$ ({n_amostras} amostras)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = eixos[0, 2]
    ax.plot(atrasos, autocorr, color="#9333EA", linewidth=2)
    ax.axhline(1 / np.e, color="gray", linestyle=":", label="1/e")
    ax.axvline(comp_corr, color="red", linestyle="--",
               label=fr"$\ell={comp_corr:.1f}$")
    ax.set_xlabel("atraso")
    ax.set_ylabel("autocorrelação")
    ax.set_title("Autocorrelação (passeio aleatório)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    for k, (gi, gj, z, ni, nj, ii, jj) in enumerate(cortes):
        ax = eixos[1, k]
        cs = ax.contourf(gi, gj, np.log10(z), levels=18, cmap="viridis")
        ax.contour(gi, gj, np.log10(z), levels=8, colors="white",
                   linewidths=0.5, alpha=0.6)
        ax.plot(THETA_REAL[ii], THETA_REAL[jj], "r*", markersize=16,
                markeredgecolor="white", label=r"$\theta^*$")
        ax.set_xlabel(ni)
        ax.set_ylabel(nj)
        ax.set_title(rf"$\log_{{10}} f$ no plano ({ni}, {nj})")
        plt.colorbar(cs, ax=ax, fraction=0.046)
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)

    faixa = f"{CENARIOS[cenario][0]*100:.0f}-{CENARIOS[cenario][1]*100:.0f}%"
    fig.suptitle(f"Análise de paisagem - cenário {NOME_CENARIO[cenario]} "
                 f"({faixa} de theta*)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    destino = PASTA / f"paisagem_{cenario}.png"
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    imprimir(f"    figura salva: {destino.name}")
    return {"FDC": fdc, "comp_corr": comp_corr, "f_real": f_real}


# ------------------------------------------------------------------------------
# 4. Experimento principal: 360 execucoes
# ------------------------------------------------------------------------------
def experimento(f):
    piso = f(THETA_REAL)
    total = N_REPETICOES * len(LISTA_CENARIOS) * len(OTIMIZADORES)
    imprimir("\n" + "=" * 72)
    imprimir(f"  [3/4]  Experimento principal - {total} execucoes "
             f"({N_REPETICOES} repeticoes x {len(LISTA_CENARIOS)} cenarios "
             f"x {len(OTIMIZADORES)} algoritmos)")
    imprimir("=" * 72)

    registros, historicos = [], {}
    inicio = time.time()
    for cen in LISTA_CENARIOS:
        lo, hi = caixa_de_busca(THETA_REAL, cen)
        imprimir(f"\n  Cenario {NOME_CENARIO[cen]}:")
        for nome, otimizador in OTIMIZADORES.items():
            valores_f = []
            for semente in range(N_REPETICOES):
                r = otimizador(f, lo, hi, MAX_AVALIACOES, semente)
                erro = mape(r["theta"])
                registros.append({
                    "cenario": cen, "algo": nome, "seed": semente,
                    "f": r["f"], "excesso": r["f"] - piso,
                    "R_est": r["theta"][0], "X_est": r["theta"][1],
                    "C_est": r["theta"][2], "G_est": r["theta"][3],
                    "MAPE_R": erro[0], "MAPE_X": erro[1], "MAPE_C": erro[2],
                    "absG": abs(r["theta"][3]), "n_eval": r["n_aval"],
                })
                valores_f.append(r["f"])
                historicos[(nome, cen, semente)] = r["historico"]
            imprimir(f"    {nome:<10} f mediano = {np.median(valores_f):.4e}   "
                     f"f minimo = {np.min(valores_f):.4e}")
    imprimir(f"\n  Tempo do experimento: {time.time()-inicio:.1f}s")

    df = pd.DataFrame(registros)
    df.to_csv(PASTA / "resultados_oficiais.csv", index=False)
    pacote = {f"{a}_{c}_{s}": h for (a, c, s), h in historicos.items()}
    np.savez_compressed(PASTA / "historicos.npz", **pacote)
    imprimir(f"  resultados_oficiais.csv ({len(df)} linhas) e historicos.npz")
    return df, historicos, piso


# ------------------------------------------------------------------------------
# 5. Analise estatistica: Friedman, Nemenyi e Wilcoxon
# ------------------------------------------------------------------------------
def diferenca_critica(k, N, alpha=0.05):
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6.0 * N))


def nemenyi_pvalores(tabela):
    if TEM_POSTHOCS:
        return sp.posthoc_nemenyi_friedman(tabela.values).values
    # Sem a biblioteca, calcula direto pela estatistica de amplitude estudentizada
    N, k = tabela.shape
    ranks = np.array([rankdata(tabela.values[i]) for i in range(N)]).mean(0)
    P = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            q = abs(ranks[i] - ranks[j]) / np.sqrt(k * (k + 1) / (6 * N))
            P[i, j] = P[j, i] = studentized_range.sf(q * np.sqrt(2), k, np.inf)
    return P


def analise_estatistica(df, historicos, piso):
    imprimir("\n" + "=" * 72)
    imprimir("  [4/4]  Analise estatistica")
    imprimir("=" * 72)

    # Tabela resumo
    linhas = []
    for cen in LISTA_CENARIOS:
        for alg in ALGORITMOS:
            sub = df[(df.cenario == cen) & (df.algo == alg)]
            linhas.append({
                "Cenario": NOME_CENARIO[cen], "Algoritmo": alg,
                "Md f": sub.f.median(),
                "IQR f": sub.f.quantile(.75) - sub.f.quantile(.25),
                "Md MAPE_R": sub.MAPE_R.median(),
                "Md MAPE_X": sub.MAPE_X.median(),
                "Md MAPE_C": sub.MAPE_C.median(),
                "Acertos/30": int((sub.excesso <= 1e-4).sum()),
            })
    resumo = pd.DataFrame(linhas)
    resumo.to_csv(PASTA / "resumo_tabela.csv", index=False)

    # Teste de Friedman
    tabela = df.pivot_table(index=["cenario", "seed"],
                            columns="algo", values="excesso")[ALGORITMOS]
    qui2, p_friedman = friedmanchisquare(*[tabela[a].values for a in ALGORITMOS])
    N, k = tabela.shape
    ranks = np.array([rankdata(tabela.values[i]) for i in range(N)]).mean(0)
    rank_algo = dict(zip(ALGORITMOS, ranks))
    imprimir(f"\n  Friedman:  qui2 = {qui2:.2f}   p = {p_friedman:.2e}   "
             f"(N={N} blocos)")
    imprimir("  Ranks medios: " +
             "  ".join(f"{a}={r:.2f}" for a, r in
                       sorted(rank_algo.items(), key=lambda kv: kv[1])))

    # Post-hoc de Nemenyi
    CD = diferenca_critica(k, N, ALPHA)
    _ = nemenyi_pvalores(tabela)
    imprimir(f"  Nemenyi:   diferenca critica = {CD:.3f}")

    # Wilcoxon par a par com correcao de Holm
    imprimir("\n  Wilcoxon par a par (correcao de Holm, 6 pares por cenario):")
    pares_wil = []
    for cen in LISTA_CENARIOS:
        sub = df[df.cenario == cen]
        pares = [(a, b) for ia, a in enumerate(ALGORITMOS)
                 for b in ALGORITMOS[ia + 1:]]
        p_bruto = []
        for a, b in pares:
            xa = sub[sub.algo == a].sort_values("seed").excesso.values
            xb = sub[sub.algo == b].sort_values("seed").excesso.values
            p_bruto.append(1.0 if np.allclose(xa, xb) else wilcoxon(xa, xb)[1])
        # correcao de Holm
        ordem = np.argsort(p_bruto)
        p_holm = np.empty(len(p_bruto))
        for posicao, idx in enumerate(ordem):
            p_holm[idx] = min(1.0, p_bruto[idx] * (len(p_bruto) - posicao))
        p_holm = np.maximum.accumulate(p_holm[ordem])[np.argsort(ordem)]
        for (a, b), pr, ph in zip(pares, p_bruto, p_holm):
            xa = sub[sub.algo == a].sort_values("seed").excesso.values
            xb = sub[sub.algo == b].sort_values("seed").excesso.values
            vencedor = a if np.median(xa) < np.median(xb) else b
            sig = "n.s." if ph > ALPHA else "***"
            pares_wil.append({"cenario": NOME_CENARIO[cen], "par": f"{a} vs {b}",
                              "p_raw": pr, "p_holm": ph,
                              "vencedor": vencedor, "sig": sig})
    pd.DataFrame(pares_wil).to_csv(PASTA / "wilcoxon_pares.csv", index=False)

    de = df[df.algo == "DE"].sort_values(["cenario", "seed"]).excesso.values
    pso = df[df.algo == "PSO"].sort_values(["cenario", "seed"]).excesso.values
    if np.allclose(de, pso):
        imprimir("    DE vs PSO: empate (p=1,0) - mesmo theta em precisao de maquina")
    else:
        imprimir("    DE vs PSO: diferem")

    # Figura: curvas de convergencia
    fig, eixos = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, cen in zip(eixos, LISTA_CENARIOS):
        for alg in ALGORITMOS:
            curvas = []
            for semente in range(N_REPETICOES):
                h = historicos[(alg, cen, semente)]
                if len(h):
                    curvas.append(h)
            malha = np.linspace(1, MAX_AVALIACOES, 100)
            interpoladas = []
            for h in curvas:
                excesso = np.maximum(h[:, 1] - piso, 1e-12)
                interpoladas.append(np.interp(malha, h[:, 0], excesso))
            interpoladas = np.array(interpoladas)
            mediana = np.median(interpoladas, axis=0)
            faixa_lo = np.percentile(interpoladas, 2.5, axis=0)
            faixa_hi = np.percentile(interpoladas, 97.5, axis=0)
            ax.plot(malha, mediana, color=COR[alg], label=alg, linewidth=1.8)
            ax.fill_between(malha, faixa_lo, faixa_hi, color=COR[alg], alpha=0.15)
        ax.set_yscale("log")
        ax.set_xlabel("Avaliações de $f$")
        faixa = f"{CENARIOS[cen][0]*100:.0f}%-{CENARIOS[cen][1]*100:.0f}%"
        ax.set_title(f"Cenário {NOME_CENARIO[cen]}  ({faixa})")
        ax.grid(alpha=0.3, which="both")
    eixos[0].set_ylabel(r"$f(\theta) - f(\theta^*)$  (excesso sobre o piso)")
    eixos[-1].legend(fontsize=9)
    fig.suptitle(f"Curvas de convergência - mediana e IC 95% (n = {N_REPETICOES})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PASTA / "curva_convergencia_oficial.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # Figura: dispersao do MAPE de X'
    fig, eixos = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, cen in zip(eixos, LISTA_CENARIOS):
        dados_cx = [df[(df.cenario == cen) & (df.algo == a)].MAPE_X.values
                    for a in ALGORITMOS]
        caixas = ax.boxplot(dados_cx, tick_labels=ALGORITMOS, patch_artist=True)
        for caixa, alg in zip(caixas["boxes"], ALGORITMOS):
            caixa.set_facecolor(COR[alg])
            caixa.set_alpha(0.6)
        ax.set_yscale("log")
        ax.set_title(f"Cenário {NOME_CENARIO[cen]}")
        ax.grid(alpha=0.3, which="both", axis="y")
    eixos[0].set_ylabel("MAPE de $X'$ [%]")
    fig.suptitle("MAPE de $X'$ por algoritmo (30 execuções por caixa)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PASTA / "boxplot_oficial.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figura: diagrama de diferenca critica
    _diagrama_cd(ranks, CD, ALGORITMOS, PASTA / "cd_diagram.png")

    imprimir("    figuras: curva_convergencia_oficial.png, "
             "boxplot_oficial.png, cd_diagram.png")

    imprimir("\n  Tabela resumo (mediana de 30 execucoes):")
    imprimir(resumo.to_string(index=False))
    return resumo, {"qui2": qui2, "p": p_friedman, "CD": CD, "ranks": rank_algo}


def _diagrama_cd(rank_medio, cd, nomes, destino):
    k = len(nomes)
    ordem = np.argsort(rank_medio)
    ranks = rank_medio[ordem]
    nomes_ord = [nomes[i] for i in ordem]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0.4, k + 0.6)
    ax.set_ylim(-2.2, 2.3)
    ax.invert_xaxis()
    ax.set_yticks([])
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    ax.set_xticks(np.arange(1, k + 1))
    ax.set_xlabel("Rank médio (1 = melhor)")
    alturas = [1.6, 0.85, 1.6, 0.85]
    for i, (r, nm) in enumerate(zip(ranks, nomes_ord)):
        h = alturas[i % len(alturas)]
        ax.plot([r, r], [0, h - 0.28], "k-", linewidth=1.1)
        ax.text(r, h, f"{nm}" + "\n" + "(" + f"{r:.2f}".replace(".", ",") + ")",
                fontsize=11, ha="center", va="bottom", linespacing=1.1)
    base_y, base_x = -1.0, ranks[0]
    ax.plot([base_x, base_x - cd], [base_y, base_y], "k-", linewidth=2.5)
    for xx in (base_x, base_x - cd):
        ax.plot([xx, xx], [base_y - 0.12, base_y + 0.12], "k-", linewidth=1.5)
    ax.text(base_x - cd / 2, base_y - 0.42,
            "CD = " + f"{cd:.3f}".replace(".", ","), ha="center", fontsize=10)
    equivalentes = [(i, j) for i in range(k) for j in range(i + 1, k)
                    if abs(ranks[i] - ranks[j]) < cd]
    if not equivalentes:
        ax.text((ranks[0] + ranks[-1]) / 2, -1.85,
                "Nenhum par equivalente - todos os algoritmos diferem (a = 0,05)",
                ha="center", fontsize=9.5, style="italic", color="gray")
    ax.set_title("Diagrama de diferença crítica de Nemenyi (a = 0,05; CD = "
                 + f"{cd:.3f}".replace(".", ",") + ")", fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------
# Execucao
# ------------------------------------------------------------------------------
def main():
    imprimir("=" * 72)
    imprimir("  Trabalho final de Metaheuristicas - Vinicius M. D. Conceicao")
    imprimir("  Estimacao de parametros de LT por metaheuristicas (PPGEE/UFMT)")
    if MODO_RAPIDO:
        imprimir("  (modo rapido: poucas repeticoes, apenas para teste)")
    imprimir("=" * 72)

    arquivo = PASTA / "dados_SNR30.csv"
    if not arquivo.exists():
        imprimir(f"\n  ERRO: 'dados_SNR30.csv' nao encontrado em {PASTA}")
        imprimir("  Coloque o arquivo na mesma pasta deste script e rode de novo.")
        sys.exit(1)

    inicio = time.time()

    imprimir("\n  [1/4]  Carregando dados e validando a funcao objetivo...")
    dados = carregar_dados(arquivo)
    f = construir_objetivo(dados)
    piso = f(THETA_REAL)
    vl, il = prever_VL_IL(THETA_REAL, dados["V0"], dados["I0"])
    f_sem_ruido = construir_objetivo({**dados, "VL": vl, "IL": il})(THETA_REAL)
    imprimir(f"    f(theta*) = {piso:.4e}   |   f sem ruido = {f_sem_ruido:.1e} "
             f"(deve ser proximo de zero)")

    analise_paisagem(f, "dificil")
    df, historicos, piso = experimento(f)
    analise_estatistica(df, historicos, piso)

    imprimir("\n" + "=" * 72)
    imprimir(f"  Concluido em {time.time()-inicio:.0f}s. Arquivos gerados em:")
    imprimir(f"    {PASTA}")
    imprimir("  Figuras: paisagem_dificil, curva_convergencia_oficial, "
             "boxplot_oficial, cd_diagram")
    imprimir("  Tabelas: resultados_oficiais, resumo_tabela, wilcoxon_pares")
    imprimir("=" * 72)


if __name__ == "__main__":
    main()
