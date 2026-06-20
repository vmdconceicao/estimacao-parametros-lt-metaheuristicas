# Estimação de Parâmetros de Linha de Transmissão por Metaheurísticas

Trabalho final da disciplina de **Metaheurísticas** (PPGEE/UFMT, 1º semestre de 2026).

**Autor:** Vinicius Marcos Domingues Conceição
**Disciplina:** Metaheurísticas — Prof. Raoni F. S. Teixeira

---

## Sobre o trabalho

Este projeto estima os quatro parâmetros elétricos de sequência positiva de uma
linha de transmissão de 200 km — resistência (R'), reatância (X'), capacitância (C')
e condutância (G') por quilômetro — a partir de medições fasoriais de tensão e
corrente contaminadas por ruído (SNR = 30 dB).

O problema é formulado como uma otimização contínua: encontrar o vetor de parâmetros
que minimiza o erro entre os fasores medidos e os previstos pelo modelo de
parâmetros distribuídos (matriz ABCD). Quatro algoritmos são comparados:

- **Quasi-Newton** (L-BFGS-B com reinícios aleatórios) — baseline clássico
- **Simulated Annealing (SA)**
- **Differential Evolution (DE)**
- **Particle Swarm Optimization (PSO)**

A comparação segue um protocolo de 30 execuções independentes por algoritmo, em
três cenários de incerteza (±5%, ±10% e ±20%), totalizando 360 execuções. Os
resultados são validados com os testes não-paramétricos de Friedman, Nemenyi e
Wilcoxon.

## Principais resultados

- **DE e PSO** atingem o melhor resultado possível (o piso imposto pelo ruído) em
  todas as 30 execuções dos três cenários.
- O **Quasi-Newton** reproduz o *reference shift problem* da literatura: sua taxa
  de acerto cai de 29/30 (cenário fácil) para 2/30 (cenário difícil).
- O erro residual de 2,86% em R' não é falha dos algoritmos, e sim o limite de
  identificabilidade imposto pelo ruído de medição (SNR = 30 dB).

## Como executar

1. Tenha o **Python 3.11 ou superior** instalado.
2. Instale as dependências:

   ```
   pip install numpy pandas scipy matplotlib scikit-posthocs
   ```

3. Mantenha o arquivo `METAHEURISTICA.py` na **mesma pasta** que `dados_SNR30.csv`.
4. Execute:

   ```
   python METAHEURISTICA.py
   ```

   Para um teste rápido com menos repetições (apenas para verificar se tudo roda):

   ```
   python METAHEURISTICA.py --rapido
   ```

O script gera automaticamente todas as figuras (análise de paisagem, curvas de
convergência, box-plots e diagrama de diferença crítica) e as tabelas de
resultados em formato CSV.

## Arquivos do repositório

| Arquivo | Descrição |
|---|---|
| `METAHEURISTICA.py` | Script único que executa todo o trabalho de ponta a ponta |
| `dados_SNR30.csv` | Conjunto de dados sintético (5000 medições, SNR = 30 dB) |
| `README.md` | Este arquivo |

### Arquivos gerados ao executar

- `paisagem_dificil.png` — análise de paisagem (FDC, autocorrelação, cortes 2D)
- `curva_convergencia_oficial.png` — curvas de convergência com IC 95%
- `boxplot_oficial.png` — dispersão do erro em X'
- `cd_diagram.png` — diagrama de diferença crítica de Nemenyi
- `resultados_oficiais.csv` — resultado bruto das 360 execuções
- `resumo_tabela.csv` — tabela resumo (mediana, IQR, MAPE, acertos)
- `wilcoxon_pares.csv` — comparações par a par de Wilcoxon

## Reprodutibilidade

Todos os resultados são determinísticos: as sementes aleatórias estão fixadas no
código (0 a 29 no experimento; 42 na análise de paisagem).

**Ambiente de referência:** Python 3.12, NumPy 1.26, SciPy 1.12,
scikit-posthocs 0.13, matplotlib 3.9. Tempo total de execução: cerca de 3 minutos.

## Declaração de uso de IA

Em conformidade com as regras da disciplina, declara-se que foi utilizada a
ferramenta de inteligência artificial generativa Claude (Anthropic) como apoio na
implementação e depuração do código, na formatação do manuscrito e na revisão de
texto. A formulação do problema, a escolha metodológica, a execução dos
experimentos, a conferência dos resultados e as conclusões são de responsabilidade
do autor, que revisou e validou todo o conteúdo. Os resultados são integralmente
reproduzíveis por meio deste repositório.
