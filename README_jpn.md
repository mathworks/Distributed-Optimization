# Distributed Optimization: Overview and MATLAB examples

## Contents

1. [Overview](#overview)

2. [Simple Example of Multi-Agent Optimization](#simple-example-of-multi-agent-optimization)

3. [Real-world Application: Distributed Resource Allocation for Electric Vehicle (EV) Charging](#real-world-application-distributed-resource-allocation-for-electric-vehicle-ev-charging)
   
    3.1. [Solve with **Distributed Gradient Descent (DGD)**](#solve-with-dgd)
   
    3.2. [Solve with **Alternating Direction Method of Multipliers (ADMM)**](#solve-with-admm)
   
    3.3. [Solve with **Particle-Swarm Optimization (PSO)**](#solve-with-pso)
   
    3.4. [Solve with **Genetic Algorithm (GA)**](#solve-with-ga)

4. [Discussion](#discussion)

5. [References](#references)

## Overview

分散最適化とは、グローバルな目的関数の一部やローカルな制約条件のみを把握している複数のエージェントが、通信ネットワークを介して情報を交換しながら協調して共同の最適化問題を解くアルゴリズムやフレームワークの総称です。中央の調整者を必要とせず、各エージェントは勾配降下法、双対分解、交互方向乗数法などを用いて局所変数を反復的に更新しつつ、コンセンサス手続きを通じて共有変数や結合制約の整合性を保ちます。この手法は問題規模に応じてスケーラブルであり、各エージェントのデータプライバシーを確保しながら、非同期更新や時間的に変動するネットワークトポロジーへの適応も可能です。センサーフュージョンやワイヤレスネットワークのリソース配分、自律走行車隊の協調制御など、通信遅延やエージェントの故障に対する耐性が求められる応用分野で広く利用されています。

### General Concept

```sql
1. Global Optimization Problem
   F(x₁, x₂, …, x_N)
   └─► N個のエージェントを協調させることによりFの最小化/最大化を目指す

2. Multi-Agent System
   ├─ Agent 1
   ├─ Agent 2
   ├─ …
   └─ Agent N

3. Agent 1 Workflow (代表例)
   ├─ [Local Data/Knowledge] (ローカルデータ/知識)
   │     • 例: p_pref₁, ローカル制約
   ├─ [Local Objective/Cost] (ローカル目的/コスト)
   │     • 例: (p₁ – p_pref₁)² または F のローカル部分
   ├─ [Local Computation Engine] (ローカル計算エンジン)
   │     • 入力: x₁(k), ローカル目的
   │     └─► Method Choice: (手法の選択)
   │          a) Gradient-Based (勾配ベース)
   │             1. ∇f₁(x₁(k)) を計算
   │             2. ∇f₁ とメッセージを使用して更新
   │          b) Gradient-Free (勾配フリー)
   │             1. f₁(x₁(k)) を評価
   │             2. ヒューリスティック更新 + メッセージ
   ├─ [Prepare & Send Message] (メッセージの準備と送信)
   │     • データは手法に依存 (状態、勾配、入札など)
   ├─ [Communication Network] (通信ネットワーク)
   │     • トポロジー: リング / スター / メッシュ / 完全グラフ
   │     • 隣接エージェント間でメッセージを運ぶ
   ├─ [Receive Messages] (メッセージの受信)
   │     • 隣接エージェントから
   ├─ [Update State] (状態の更新)
   │     • 新しい推定値 x₁(k+1)
   └─ [Iterate k→k+1] (反復 k→k+1)

4. Agent 2 Workflow
   (Agent 1 と同じステップ、自身のデータ/目的を使用)

5. …

6. Agent N Workflow
   (同じパターン)

7. What’s Exchanged on the Network (ネットワーク上で交換されるもの)
   – Current states/estimates x_j(k) (現在の状態/推定値 x_j(k))
   – Local gradients ∇f_j (ローカル勾配 ∇f_j)
   – Dual variables / prices λ_j (双対変数 / 価格 λ_j)
   – Proposed actions or bids (提案されたアクションまたは入札)
   – Best-known solutions (for meta-heuristics) (既知の最良解 (メタヒューリスティクス用))
   – Aggregated values (sums, averages) (集計値 (合計、平均))
```

**図とマルチエージェント最適化プロセスの説明:**

1. **Global Optimization Problem (F(x_1, ..., x_N)):**
   
   * これは、エージェントのシステム全体が達成しようとする包括的な目標です。
   * 多くの場合、すべてのエージェントの変数/アクションの関数です (例: 総コストの最小化、`sum f_i(x_i)`、またはコンセンサス `x_1 = x_2 = ... = x_N` の達成)。
   * エージェントの最適なアクションが他のエージェントのアクションや共有制約に依存する場合があるため、解決にはしばしば調整が必要です。

2. **Multi-Agent System:**
   
   * 複数の自律的な `Agent` (Agent 1, Agent 2, ..., Agent N) で構成されます。
   * 各エージェント `i` は、独自の決定変数 `x_i` を持ちます。

3. **Individual Agent (e.g., Agent 1):**
   
   * **Local Data/Knowledge (ローカルデータ/知識):** 各エージェントはプライベートな情報を持っています (例: 優先充電レート `p_pref_i`、ローカルセンサーの読み取り値、ローカル制約)。
   * **Local Objective/Cost (ローカル目的/コスト):** ローカルデータとグローバルな目的に基づいて、各エージェントはしばしば最適化するためのローカル目的関数 `f_i(x_i)` を持つか、グローバルな目的に一部貢献します。時には、ローカル目的には、結合や合意を強制するために隣人の状態に関連する項も含まれます。
   * **Agent State/Estimate (x_i(k)) (エージェントの状態/推定値):** 各反復 `k` で、エージェントは決定変数の現在の値またはグローバルな量の推定値を維持します。
   * **Local Computation Engine (ローカル計算エンジン):** これはエージェントの意思決定の中核です。
     * 現在の状態 `x_i(k)`、ローカル目的 `f_i`、ローカルデータ、および他のエージェントから受信したメッセージを入力として受け取ります。
     * 次に、`Local Method` (ローカル手法) を適用して、次のアクションを計算するか、状態を更新します。
   * **Local Method (ローカル手法):**
     * **Gradient-Based (勾配ベース):**
       * ローカル目的 `f_i` が微分可能な場合、エージェントはローカル勾配 `∇f_i(x_i(k))` を計算できます。
       * 更新ルールは通常、この負の勾配に沿ったステップ (最小化の場合) と隣人からの情報 (例: DGDにおける隣人の状態の加重平均、またはADMMにおける双対変数に基づく更新) を組み合わせます。
       * 例: 分散勾配降下法 (DGD)、ADMMサブプロブレム、双対分解法。
     * **Gradient-Free (勾配フリー):**
       * 勾配が利用できない、計算コストが高すぎる、または問題が微分不可能またはブラックボックスである場合に使用されます。
       * エージェントは、現在の状態 (または提案された状態) でローカル目的 `f_i(x_i(k))` を評価します。
       * 更新ルールは、特定のアルゴリズムから導出されたヒューリスティックな原則に従います (例: PSO粒子は、自身のパーソナルベストと近傍/グローバルベストに基づいて速度と位置を更新します。GA個体は選択、交叉、突然変異を受けます)。
       * 例: 粒子群最適化 (PSO)、遺伝的アルゴリズム (GA)、アリコロニー最適化 (ACO)、シミュレーテッドアニーリング (分散設定に適合させた場合)。
   * **Prepare/Send Message (メッセージの準備と送信):** ローカル計算の後、エージェントは隣人に送信するための関連情報を含むメッセージを準備します。
   * **Receive Messages (メッセージの受信):** 次の計算サイクルの前に、エージェントは隣人からメッセージを受信します。
   * **New State/Estimate (x_i(k+1)) (新しい状態/推定値):** ローカル計算と受信したメッセージに基づいて、エージェントは状態を更新します。
   * **Iterate (k -> k+1) (反復):** 設定された反復回数または収束基準が満たされるまで、プロセスが繰り返されます。1回の反復で受信されたメッセージは、通常、*次*の反復の計算に使用されます。

4. **Communication Network (通信ネットワーク):**
   
   * どのエージェントが互いにメッセージを交換できるかを定義します。
   * **Topology (トポロジー):** 完全グラフ (全対全)、リング、スター、メッシュ、またはその他のグラフ構造にすることができます。トポロジーは、収束速度と堅牢性に大きな影響を与えます。
   * 通信は通常、不完全であると想定されます (より高度なモデルでは、遅延、ノイズ、リンク障害)。ここでは簡単のため、信頼性の高い隣人間通信を想定しています。

5. **Information Exchanged (What is Communicated) (交換される情報):**
   
   * メッセージの内容は、特定の分散アルゴリズムに大きく依存します。
     * **Current states/estimates (x_j(k)) (現在の状態/推定値):** コンセンサスアルゴリズムで一般的です (DGDのように、エージェントが隣人の状態を平均化する場合など)。
     * **Local gradients (∇f_j) (ローカル勾配):** 一部のアルゴリズムでは勾配情報を交換する場合があります。
     * **Dual variables / Prices (λ_j) (双対変数 / 価格):** ADMMおよび双対分解法の中核であり、共有制約に違反することの「価格」を表します。
     * **Proposed actions / Bids (提案されたアクション / 入札):** 市場ベースのメカニズムや一部のゲーム理論的アプローチで使用されます。
     * **Best-known solutions (既知の最良解):** 一部の分散メタヒューリスティクスで使用されます (例: PSOの粒子が自身のパーソナルベスト位置を共有したり、近傍のベストに関する情報を受信したりする)。
     * **Aggregated values (集計値):** エージェントは、特定のローカル値の合計や平均を協調して計算する場合があります (例: ADMM EV充電の例における `sum(p_i)` はコンセンサスを介して推定されました)。

**Overall Flow (全体的な流れ):**

システムは通常、反復的なラウンドで動作します。

1. **Initialization (初期化):** エージェントは初期状態 `x_i(0)` から開始します。
2. **Iteration `k` (反復 `k`):**
   a.  **Local Computation (ローカル計算):** 各エージェント `i` は、現在の状態 `x_i(k)` および前のラウンド (または反復 `k-1` から) で受信したメッセージを使用して、選択した手法 (勾配ベースまたは勾配フリー) に従ってローカル計算を実行します。
   b.  **State Update (状態更新):** エージェントは状態を `x_i(k+1)` に更新します。
   c.  **Communication (通信):** 各エージェント `i` は、メッセージ (`x_i(k+1)` の一部またはその他の派生情報を含む) を通信ネットワークを介して隣人に送信します。
3. **Repeat (繰り返し):** `k` をインクリメントし、収束または最大反復回数に達するまでステップ2aに進みます。

目標は、エージェントの状態 `x_i` が集合的にグローバルな最適化問題 `F` の解 (または良好な近似解) に収束することです。

### Foundations of Distributed Optimization

- **Consensus‐based Gradient Methods** 
  
  - **Distributed Gradient Descent (DGD)** およびその高速化版（例：EXTRA, DIGing, NIDS）は、ローカルな勾配ステップとコンセンサス（平均化）を組み合わせることで、ネットワーク上の（強）凸問題を解きます。
  
  - **Gradient Tracking** メソッドは、各エージェントでグローバルな勾配推定値を追跡し、より速い収束と時変グラフに対するロバスト性をもたらします。

- **Operator Splitting & ADMM** 
  
  - **Decentralized ADMM** は、交互方向乗数法 (Alternating Direction Method of Multipliers) をピアツーピア方式で適用し、グローバルな目的関数をローカルなサブ問題とコンセンサス制約に分割します。
  
  - 最近のバリアントは、メッセージングのオーバーヘッドを削減するために、**非同期 (asynchronous)** 、**イベントトリガー型 (event‐triggered)** 、または **通信圧縮型 (communication‐compressed)** の更新に焦点を当てています。

- **Primal–Dual & Second-Order Methods** 
  
  - **Primal–Dual Algorithms** （例：Chambolle–Pock スタイル）はネットワークに適応されており、鞍点問題に対する高速な収束率を提供します。
  
  - **Distributed Newton / Quasi-Newton** メソッドは、局所的に曲率情報を近似することで、悪条件の問題に対する収束を高速化し、多くの場合、追加の計算と引き換えに通信ラウンド数を削減します。

---

### Swarm Intelligence Meta-heuristics

- **Particle Swarm Optimization (PSO) and Variants** 
  
  - **Quantum-behaved PSO (QPSO)** 、**Bare-Bones PSO** 、および **Fully Informed PSO** は、探索と活用のトレードオフを改善します。
  
  - **Multi‐swarms** は、大規模な探索を相互作用するサブスウォームに分解し、各サブスウォームが問題の一部を最適化したり、異なる領域を探索したりします。

- **Ant Colony Optimization (ACO)** 
  
  - **Max–Min ACO** は、早期収束を避けるためにフェロモンに明示的な境界を設定します。
  
  - **Hybrid ACO-PSO** や **ACO-GA** は、相補的なメカニズム（交叉/突然変異または速度更新）を利用して局所最適解から脱出します。

- **Other Bio-inspired Techniques** 
  
  - **Artificial Bee Colony (ABC)** 、**Bacterial Foraging Optimization (BFO)** 、および **Firefly Algorithm** は、動的なパラメータ適応、並列集団、およびレヴィフライトステップによって調整され、高次元で多峰性の目的関数に対するパフォーマンスを向上させます。

---

### Swarm / Multi-Agent Control

- **Consensus & Formation Control** 
  
  - **Leaderless Consensus** プロトコルは、ローカルな隣接エージェントとの通信のみにもかかわらず、すべてのエージェントが状態（例：フォーメーションの重心、ランデブーポイント）について合意することを保証します。
  
  - **Leader–Follower** アーキテクチャは、全体的な動きを操縦する情報を持つ「リーダー」エージェントのサブセットと、形状を維持するためにコンセンサスを実行するフォロワーを組み合わせます。

- **Distributed Model Predictive Control (DMPC)** 
  
  - エージェントは、ADMM や双対分解を介して結合制約（例：衝突回避、リソース共有）を持つローカルな MPC 問題を解き、限られた通信で集中型に近いパフォーマンスを達成します。

- **Event-Triggered & Self-Triggered Control** 
  
  - ネットワーク負荷を軽減するために、エージェントは特定状態またはエラーベースの閾値を超えた場合にのみ情報を交換し、大幅に少ないメッセージで安定性とパフォーマンスを保証します。

- **Learning-Enhanced Control** 
  
  - **Graph Neural Networks (GNNs)** および **Multi-Agent Reinforcement Learning (MARL)** は、データから直接制御ポリシーを調整することを学習し、非線形ダイナミクスや部分的観測可能性に対応します。

---

### Key Trends & Open Challenges

- **Communication Efficiency** : 帯域幅を削減するための圧縮メッセージ、量子化更新、および散発的/イベントトリガー型スキーム。

- **Robustness & Privacy** : リンク障害、敵対的エージェントを許容し、差分プライバシーを保証するアルゴリズム。

- **Scalability** : 階層的スウォーム、部分空間分解、または平均場近似を介した数千のエージェントの処理。

- **Nonconvex & Time-Varying Objectives** : 非凸ランドスケープ（ロボティクスや深層学習で一般的）および動的に変化するネットワークトポロジーにおける収束の保証。

- **Bridging Meta-heuristics & Convex Methods** : より良いグローバル探索のために、高速なコンセンサスベースの凸ソルバーをグローバルなメタヒューリスティックの外側ループ内に埋め込むハイブリッドフレームワーク。

---

### In Summary

マルチエージェントおよびスウォームコンテキストにおける最先端の分散最適化は、（証明可能な保証を持つ）**厳密な凸/ADMMスタイルアルゴリズム**と、（困難で非凸なランドスケープのための）**生物に着想を得たメタヒューリスティクス**を相乗させ、現代のスウォーム制御は**学習駆動型**、**通信疎**、かつ**ロバスト**な協調法則を推進しています。この融合により、大規模なロボットチーム、センサーネットワーク、および分散機械学習システムが、現実的なネットワークおよびエージェントの制約下で効率的に動作することが可能になります。

## Simple Example of Multi-Agent Optimization

以下は、分散最適化における古典的な**平均合意 (average consensus)** のトイプロブレムを示すプロトタイプの MATLAB スクリプトです。エージェントは、接続された通信グラフ上で**ローカル勾配ステップ**と**コンセンサス平均化**を交互に行うことにより、二次コスト

$$
f(x)\;=\;\sum_{i=1}^N \tfrac12\bigl(x - a_i\bigr)^2 
$$

を協調して最小化します。

---

### Problem Formulation

分散最適化における標準的なトイプロブレムの1つは、**平均合意 (average consensus)** 問題です。この問題では、$N$ 個のエージェントが、ローカル通信のみを使用して初期値の平均について合意することを目指します [Diva Portal](https://www.diva-portal.org/smash/get/diva2%3A453798/FULLTEXT01.pdf)。あるいは、これは二次コスト

$$
f(x)\;=\;\sum_{i=1}^N \frac12\,(x - a_i)^2, 
$$

を最小化すると見なすこともでき、その唯一の最小化子は $\{a_i\}$ の平均です [ResearchGate](https://www.researchgate.net/publication/224371202_Distributed_Subgradient_Methods_for_Multi-Agent_Optimization)。
分散（劣）勾配法は、勾配ステップとローカル平均化を組み合わせ、**分散勾配降下法 (Distributed Gradient Descent - DGD)** の反復式

$$
x_i^{(k+1)} \;=\;\sum_{j=1}^N W_{ij}\,x_j^{(k)} \;-\;\alpha\;\bigl(x_i^{(k)} - a_i\bigr), 
$$

を導きます。ここで、$W$ はコンセンサス重み行列であり、$\alpha>0$ はステップサイズです [SCIRP](https://www.scirp.org/%28S%28czeh2tfqw2orz553k1w0r45%29%29/reference/referencespapers?referenceid=3220708)。減少するステップサイズまたは小さな定数ステップサイズの下での収束はよく理解されています [Massachusetts Institute of Technology](https://web.mit.edu/Asuman/Desktop/asuman/www/documents/distributed-journal-final.pdf)。
勾配スキームを超えて、**Consensus ADMM** は、拡張ラグランジアンを介してローカル最小化とコンセンサス制約を分離することにより、このような問題に対する演算子分割アプローチを提供します [Stanford University](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)。**Consensus ALADIN** のような高度なフレームワークは、二次情報を使用して非凸および多層設定にこれらのアイデアを拡張します [arXiv](https://arxiv.org/pdf/2306.05662)。他のバリアントには、トイ逆問題における**Consensus Equilibrium** メソッド [Purdue Engineering](https://engineering.purdue.edu/~bouman/Plug-and-Play/webdocs/SIIMS01.pdf)、通信を削減するための**イベントトリガー型 (event‐triggered)** 更新 [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006925)、スマートグリッドにおける多目的最適化のための**多層コンセンサス (multi‐layer consensus)** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306261921007923) などがあります。

---

### MATLAB Prototype Script

```matlab
% Distributed Average Consensus via DGD
%-------------------------------------------------------
% Number of agents
% エージェント数
N = 10;

% Local data: each agent i holds value a(i)
% ローカルデータ: 各エージェントiは値a(i)を保持
a = randn(N,1) * 5 + 10;    % e.g. randomly around 10 (例: 10周辺のランダムな値)

% Communication graph: here, a ring or complete graph
% For simplicity, use complete graph weights (equal averaging)
% 通信グラフ: ここではリンググラフまたは完全グラフ
% 簡単のため、完全グラフの重み（等しい平均化）を使用
W = ones(N) / N;

% Stepsize (small constant for stability)
% ステップサイズ（安定性のための小さな定数）
alpha = 0.1;

% Initialization: each agent starts from zero
% 初期化: 各エージェントはゼロから開始
x = zeros(N,1);

% Number of iterations
% 反復回数
T = 100;

% Store history for plotting
% プロット用の履歴を保存
Xhist = zeros(N, T+1);
Xhist(:,1) = x;

for k = 1:T
    % Local consensus + gradient step
    % ローカルコンセンサス + 勾配ステップ
    x = W * x - alpha * (x - a);

    % Save
    % 保存
    Xhist(:,k+1) = x;
end

% True consensus value
% 真の合意値
x_star = mean(a);

% Plot convergence of each agent
% 各エージェントの収束をプロット
figure;
plot(0:T, Xhist', 'LineWidth', 1.2);
hold on;
yline(x_star, 'k--', 'LineWidth', 2);
xlabel('Iteration k');
ylabel('x_i^{(k)}');
title('Distributed Gradient Descent: Average Consensus');
legendStrings = arrayfun(@(i) sprintf('Agent %d',i), 1:N, 'UniformOutput', false);
legend([legendStrings, {'Optimal Mean'}],'Location','best');
grid on;
```

---

### How It Works

2. **Initialization** 

各エージェント $i$ は、自身の推定値 $x_i^{(0)}=0$ を初期化し、自身のプライベートデータ $a_i$ のみを知っています。

4. **Consensus Step** 

各反復において、エージェントは行列 $W$ を介して自身の状態を隣人の状態の加重平均で置き換えます。**完全グラフ (complete graph)** では、$W_{ij}=1/N$ です。

6. **Local Gradient Step** 

各エージェントは、自身の最小化子 $a_i$ に向かって $-\alpha(x_i - a_i)$ だけ調整します。

8. **Convergence** 

$\alpha$ と $W$ の接続性に関する緩やかな条件下で、すべての $x_i^{(k)}$ はグローバルな最小化子

$$
\overline{a} = \tfrac1N\sum_i a_i
$$

に収束します。

---

## Real-world Application: Distributed Resource Allocation for Electric Vehicle (EV) Charging

**Scenario:**
`N` 台の電気自動車 (EV) の集合が、総電力容量 `P_total` の充電ステーションから充電したいとします。各EV `i` は、優先充電レート `p_pref_i` を持っています（例：バッテリーサイズ、現在の充電状態、希望充電時間に基づく）。しかし、優先レートの合計が `P_total` を超える可能性があります。

**Optimization Goal:**
各EVに対して実際の充電レート `p_i` を見つけたいとします。その際、以下の条件を満たすようにします。

1. 使用される総電力が `P_total` を超えないこと: `sum(p_i) = P_total`。(利用可能な全電力を使用することを目指します)。
2. 配分が「公平」であること。これを達成する一般的な方法は、総電力制約の下で、優先レートからの二乗差の合計を最小化することです。
   Minimize `sum_i (p_i - p_pref_i)^2`
   Subject to `sum_i p_i = P_total`

**Distributed Solution Approach:**
この制約付き最適化問題の解析解は次のとおりです。
`p_i_optimal = p_pref_i + (P_total - sum(p_pref_j)) / N`

項 `(P_total - sum(p_pref_j)) / N` に注目してください。これは `P_total/N - mean(p_pref_j)` と書き換えることができます。
したがって、`p_i_optimal = p_pref_i + P_total/N - mean(p_pref)` となります。

これを分散的に実装するには：

1. 各EVは自身の `p_pref_i` とグローバルな値 `P_total` および `N` を知っています。
2. EVは協調して `mean(p_pref)` を計算する必要があります。これはまさに分散平均合意 (DGDを使用) アルゴリズムが行うことです！
3. 各EV `i` が `mean(p_pref)` の良い推定値（その推定値を `z_i` としましょう）を得たら、最適な電力配分を計算できます: `p_i = p_pref_i + P_total/N - z_i`。

```matlab
Preferred charging rates (kW):
    6.3362    8.7626    3.0009    5.4187    4.1740    3.7387    4.4901    5.7645    6.1741    7.3105

Sum of preferred rates: 55.17 kW
Total available power: 50.00 kW
```

### Solve with DGD

単純な DGD コードを適応させて、エージェントが `mean(p_pref)` に収束するようにします。

```matlab
% Distributed Resource Allocation for EV Charging via DGD
%-----------------------------------------------------------------

% Number of EVs (agents)
% EVの数（エージェント）
N = 10;

% Total Power Capacity of the charging station (e.g., kW)
% 充電ステーションの総電力容量（例：kW）
P_total = 50; 

% Local data: Each EV i has a preferred charging rate p_pref(i)
% Let's say preferred rates are between 3 kW and 11 kW (typical Level 2 charging)
% ローカルデータ：各EV i は優先充電レート p_pref(i) を持つ
% 優先レートは3kWから11kWの間と仮定（一般的なレベル2充電）
p_pref = rand(N,1) * 8 + 3; % Randomly between 3 and 11 kW (3から11kWの間でランダム)

fprintf('Preferred charging rates (kW):\n');
disp(p_pref');
fprintf('Sum of preferred rates: %.2f kW\n', sum(p_pref));
fprintf('Total available power: %.2f kW\n\n', P_total);

% Communication graph: Ring topology for a slightly more realistic scenario
% than complete graph, but still simple.
% For a ring, each agent communicates with two neighbors.
% 通信グラフ：完全グラフよりも少し現実的なシナリオのためのリングトポロジー、ただし単純。
% リングの場合、各エージェントは2つの隣接エージェントと通信する。
W = zeros(N,N);
for i = 1:N
    W(i,i) = 1/3; % Weight for self (自身への重み)
    W(i, mod(i-1-1, N)+1) = 1/3; % Weight for previous neighbor (with wrap-around) (前の隣人への重み（ラップアラウンドあり）)
    W(i, mod(i, N)+1) = 1/3;     % Weight for next neighbor (with wrap-around) (次の隣人への重み（ラップアラウンドあり）)
end
% A simple check for doubly stochastic (not strictly required for DGD convergence to mean
% for this specific formulation, but good practice for consensus)
% For DGD to average a_i, W must be doubly stochastic.
% The DGD update x = Wx - alpha*(x-a) can be x_new(i) = x_old(i) + sum_j W(i,j)*(x_old(j)-x_old(i)) - alpha*(x_old(i)-a(i))
% Or, as used: x_i^{k+1} = sum_j W_ij x_j^k - alpha * (x_i^k - a_i)
% If W is column stochastic (sum(W(:,j))=1), Wx averages. If W is row stochastic (sum(W(i,:))=1), Wx convex combs.
% For this problem, we need W to be such that W^k converges to (1/N)*ones(N,N) for consensus.
% Let's use a common Metropolis-Hastings rule for W if graph is not complete.
% Or, for simplicity and to keep it close to original, use complete graph avg.
% 二重確率性の簡単なチェック（この特定の定式化ではDGDが平均に収束するために厳密には要求されないが、コンセンサスのためには良い習慣）
% DGDがa_iを平均化するためには、Wは二重確率でなければならない。
% DGD更新 x = Wx - alpha*(x-a) は x_new(i) = x_old(i) + sum_j W(i,j)*(x_old(j)-x_old(i)) - alpha*(x_old(i)-a(i)) と書ける。
% または、使用されているように: x_i^{k+1} = sum_j W_ij x_j^k - alpha * (x_i^k - a_i)
% Wが列確率的（sum(W(:,j))=1）なら、Wxは平均化する。Wが行確率的（sum(W(i,:))=1）なら、Wxは凸結合する。
% この問題では、コンセンサスのためにW^kが(1/N)*ones(N,N)に収束するようなWが必要。
% グラフが完全でない場合は、一般的なMetropolis-HastingsルールをWに使用する。
% あるいは、簡単のため、またオリジナルに近づけるため、完全グラフの平均を使用する。
fprintf('Using simplified complete graph weights for W (equal averaging).\n');
W = ones(N) / N; 

% Stepsize for DGD (small constant for stability)
% DGDのステップサイズ（安定性のための小さな定数）
alpha = 0.2; 

% Initialization: Each agent's estimate of mean(p_pref) starts from zero
% (or could start from its own p_pref_i)
% 初期化：各エージェントのmean(p_pref)の推定値はゼロから始まる
% （または自身のp_pref_iから始めることもできる）
z = zeros(N,1); % z_i will be agent i's estimate of mean(p_pref) (z_iはエージェントiのmean(p_pref)の推定値)

% Number of iterations for consensus
% コンセンサスのための反復回数
T = 200;

% Store history for plotting convergence of z
% zの収束をプロットするための履歴を保存
Zhist = zeros(N, T+1);
Zhist(:,1) = z;

% --- DGD for estimating mean(p_pref) ---
% --- mean(p_pref)を推定するためのDGD ---
for k = 1:T
    % Each agent updates its estimate z_i based on neighbors' estimates
    % and its own preferred rate p_pref_i.
    % The DGD update: z_new = W * z_old - alpha * (z_old - p_pref)
    % Here, 'a' from the original example is 'p_pref'.
    % 'x' from the original example is 'z'.
    % 各エージェントは、隣人の推定値と自身の優先レートp_pref_iに基づいて推定値z_iを更新する。
    % DGD更新式：z_new = W * z_old - alpha * (z_old - p_pref)
    % ここで、元の例の'a'は'p_pref'。
    % 元の例の'x'は'z'。
    z = W * z - alpha * (z - p_pref); 

    Zhist(:,k+1) = z;
end

% True mean of preferred rates
% 優先レートの真の平均
mean_p_pref_true = mean(p_pref);

% Plot convergence of z_i to mean(p_pref)
% z_iのmean(p_pref)への収束をプロット
figure;
plot(0:T, Zhist', 'LineWidth', 1.2);
hold on;
yline(mean_p_pref_true, 'k--', 'LineWidth', 2);
xlabel('Iteration k');
ylabel('z_i^{(k)} (Estimate of mean p_{pref})');
title('DGD: Estimating Average Preferred Charging Rate');
legendStrings = arrayfun(@(i) sprintf('EV %d',i), 1:N, 'UniformOutput', false);
legend([legendStrings, {'True Mean p_{pref}'}],'Location','best');
grid on;

% --- Calculate final allocated power ---
% --- 最終的な割り当て電力を計算 ---
% Each EV uses its final estimate of mean(p_pref) (which is z(:,T+1))
% If consensus is perfect, all z_i(T+1) are close to mean_p_pref_true.
% 各EVはmean(p_pref)の最終推定値（z(:,T+1)）を使用する。
% コンセンサスが完璧であれば、すべてのz_i(T+1)はmean_p_pref_trueに近い。
estimated_mean_p_pref = Zhist(:, T+1); % Each EV has its own estimate (各EVは自身の推定値を持つ)

% Optimal allocation formula: p_i = p_pref_i + P_total/N - estimated_mean_p_pref_i
% 最適配分式：p_i = p_pref_i + P_total/N - estimated_mean_p_pref_i
p_allocated = p_pref + (P_total/N) - estimated_mean_p_pref;

% Ensure non-negativity (cannot draw negative power in this simple model)
% and also that no EV exceeds a maximum possible rate (e.g., its p_pref_i or a physical limit like 11kW)
% For simplicity, we'll just do non-negativity.
% More advanced: if an EV hits 0, its demand is removed and power redistributed among others.
% 非負性を保証（この単純なモデルでは負の電力を引くことはできない）
% また、どのEVも最大可能レート（例：自身のp_pref_iや11kWのような物理的制限）を超えないようにする。
% 簡単のため、非負性のみを考慮する。
% より高度な場合：EVが0になった場合、その需要は削除され、電力は他のEV間で再配分される。
p_allocated = max(0, p_allocated);

% Due to max(0,.) constraint, sum(p_allocated) might not be P_total.
% A simple proportional rescaling if total allocated power is not P_total.
% This is a heuristic step after the distributed calculation.
% A fully distributed scheme for exact sum constraint with bounds is more complex (e.g., using ADMM).
% max(0,.)制約のため、sum(p_allocated)がP_totalにならない場合がある。
% 割り当てられた総電力がP_totalでない場合、単純な比例再スケーリングを行う。
% これは分散計算後のヒューリスティックなステップ。
% 境界付きの正確な合計制約のための完全分散スキームはより複雑（例：ADMMを使用）。
current_sum_p_allocated = sum(p_allocated);
if current_sum_p_allocated > 0 && abs(current_sum_p_allocated - P_total) > 1e-3 % Add tolerance (許容誤差を追加)
    fprintf('Sum of p_allocated before scaling: %.2f kW. Rescaling to match P_total = %.2f kW.\n', current_sum_p_allocated, P_total);
    p_allocated = p_allocated * (P_total / current_sum_p_allocated);
    % Re-check for max(0,...) or other individual constraints if scaling pushed values out of bounds.
    % For this example, we'll assume this simple scaling is sufficient.
    % スケーリングによって値が境界外に出た場合は、max(0,...)や他の個別制約を再確認する。
    % この例では、この単純なスケーリングで十分であると仮定する。
    p_allocated = max(0, p_allocated); % Ensure non-negativity again after scaling (スケーリング後に再度非負性を保証)
    % If scaling makes sum too low because some hit zero, another pass might be needed.
    % For simplicity, we stop here.
    % スケーリングによって一部がゼロになり合計が低くなりすぎる場合は、別のパスが必要になる場合がある。
    % 簡単のため、ここでは停止する。
else
    fprintf('Sum of p_allocated after max(0,.): %.2f kW. Close enough to P_total = %.2f kW.\n', current_sum_p_allocated, P_total);
end


fprintf('\n--- Final Allocations ---\n');
disp('EV # | Preferred (kW) | Allocated (kW) | Estimate of Mean Pref (kW)');
for i=1:N
    fprintf('%4d | %14.2f | %14.2f | %20.2f\n', i, p_pref(i), p_allocated(i), estimated_mean_p_pref(i));
end
fprintf('--------------------------------------------------------------------\n');
fprintf('Sum of Preferred Rates: %.2f kW\n', sum(p_pref));
fprintf('Sum of Allocated Rates: %.2f kW (Target: %.2f kW)\n', sum(p_allocated), P_total);

% Analytical optimal solution (assuming perfect knowledge, no non-negativity initially)
% 解析的な最適解（完全な知識を仮定し、最初は非負性を考慮しない）
p_optimal_analytical_no_bounds = p_pref + (P_total - sum(p_pref)) / N;
% This analytical solution doesn't enforce p_i >= 0.
% True constrained optimum is harder to write in one line if bounds are active.
% この解析解はp_i >= 0を強制しない。
% 境界がアクティブな場合、真の制約付き最適解を一行で書くのは難しい。

% Plot results
% 結果をプロット
figure;
bar_data = [p_pref, p_allocated, p_optimal_analytical_no_bounds];
% Cap negative analytical results at 0 for fairer comparison in plot
% プロットでの公平な比較のため、負の解析結果を0でキャップする
bar_data(:,3) = max(0, bar_data(:,3)); 

b = bar(1:N, bar_data, 'grouped');
xlabel('Electric Vehicle (Agent)');
ylabel('Charging Rate (kW)');
title('Preferred vs. Allocated Charging Rates');
xticks(1:N);
legend('Preferred Rate', 'DGD Allocated Rate', 'Analytical Optimal (w/o bounds, then >=0)');
grid on;

if sum(p_pref) > P_total
    annotation('textbox', [0.6, 0.2, 0.1, 0.1], 'String', ...
        sprintf('Demand (%.1fkW) > Supply (%.1fkW)\nRates were scaled down.', sum(p_pref), P_total), ...
        'FitBoxToText','on', 'BackgroundColor', 'yellow');
elseif sum(p_pref) < P_total
    annotation('textbox', [0.6, 0.2, 0.1, 0.1], 'String', ...
        sprintf('Demand (%.1fkW) < Supply (%.1fkW)\nRates were scaled up.', sum(p_pref), P_total), ...
        'FitBoxToText','on', 'BackgroundColor', 'cyan');
else
    annotation('textbox', [0.6, 0.2, 0.1, 0.1], 'String', ...
        sprintf('Demand (%.1fkW) == Supply (%.1fkW)', sum(p_pref), P_total), ...
        'FitBoxToText','on', 'BackgroundColor', 'green');
end
```

**Explanation of Changes:**

1. **Problem Context:** 抽象的な値 `a(i)` から `p_pref(i)` (EVの優先充電レート) に切り替えました。`P_total` (総電力容量) を導入しました。
2. **Optimization Goal:** エージェント (EV) は集合的に、`sum p_i = P_total` の制約下で `sum (p_i - p_pref_i)^2` を最小化する配分 `p_i` を達成しようとします。
3. **Distributed Computation:**
   * 中核となるDGDアルゴリズムは、分散的に `mean(p_pref)` を計算するために使用されます。各エージェント `i` は推定値 `z_i` を維持し、これは `mean(p_pref)` に収束します。
   * DGDの更新 `z = W * z - alpha * (z - p_pref)` は、元の `x = W * x - alpha * (x - a)` と類似しています。
4. **Final Allocation:** 各エージェント (`z_i` として) によって `mean(p_pref)` が推定されると、リソースの自身のシェアを計算します: `p_i = p_pref_i + P_total/N - z_i`。
5. **Constraints Handling (Simplified):**
   * `p_i >= 0`: 非負の充電レートを保証するために `max(0, ...)` を追加しました。
   * `sum(p_i) = P_total`: `max(0,...)` の後の合計が `P_total` と一致しない場合に、単純な比例再スケーリングヒューリスティックを追加しました。これを境界を尊重しながら正確に処理する完全分散メソッドはより複雑です (例: ADMM (Alternating Direction Method of Multipliers) やより洗練された双対分解法) が、この例は中核的なアイデアを示しています。
6. **Communication Graph `W`:** 当初はリングを提案しましたが、簡単さと元の例の精神 (ここで `W=ones(N)/N` は完全グラフまたはよく調整されたコンセンサスプロトコルを意味します) との一貫性のために、`W = ones(N)/N` に戻しました。リングや他の疎なトポロジーは収束を遅くし、`W` は正しい平均への収束を保証するために慎重に選択する必要があるでしょう (例: Metropolis-Hastings重み)。
7. **Plotting:**
   * 個々の推定値 `z_i` の真の `mean(p_pref)` への収束を示します。
   * 優先レート、DGDで割り当てられたレート、および解析的な (理想的だが、最初は非負性の境界を考慮しない) 解を比較する棒グラフを示します。

この例は、基本的な分散合意アルゴリズム (DGD) が、より複雑で現実的なマルチエージェント最適化問題を解決するための構成要素となり得ることを示しています。「最適化」とは公正な配分を見つけることであり、DGDは必要なグローバルな情報 (`mean(p_pref)`) を分散型で得るのに役立ちます。

### 結果

```matlab
Using simplified complete graph weights for W (equal averaging).
Sum of p_allocated after max(0,.): 50.00 kW. Close enough to P_total = 50.00 kW.

Final Allocations ---
EV # | Preferred (kW) | Allocated (kW) | Estimate of Mean Pref (kW)
   1 |           6.34 |           5.68 |                 5.65
   2 |           8.76 |           7.70 |                 6.06
   3 |           3.00 |           2.90 |                 5.10
   4 |           5.42 |           4.92 |                 5.50
   5 |           4.17 |           3.88 |                 5.29
   6 |           3.74 |           3.52 |                 5.22
   7 |           4.49 |           4.14 |                 5.35
   8 |           5.76 |           5.21 |                 5.56
   9 |           6.17 |           5.55 |                 5.63
  10 |           7.31 |           6.49 |                 5.82
--------------------------------------------------------------------
Sum of Preferred Rates: 55.17 kW
Sum of Allocated Rates: 50.00 kW (Target: 50.00 kW)
```

![](dgd_iter.png)

![](dgd.png)

---

次に、EV充電の例を拡張して、より洗練された分散最適化手法を導入します。

## Alternating Direction Method of Multipliers (ADMM)

ADMMは、コンセンサス要件や共有制約によって結合された、より小さなサブ問題に分解できる問題に適しています。特に制約付き問題に対しては、単純な双対勾配法よりも優れた収束特性を示すことがよくあります。

**Problem Recap:**
Minimize `sum_i (p_i - p_pref_i)^2`
Subject to:

1. `sum_i p_i = P_total` (total power constraint)
2. `p_i >= 0` (non-negative power allocation)

**ADMM Approach for this Problem (Distributed Price Coordination):**

リソース割り当て問題に対して一般的なADMM定式化を使用します。各エージェント `i` (EV) は、自身の選好と「価格」シグナルに基づいてローカルな問題を解きます。その後、エージェントは協調してこの価格を更新し、グローバルな電力制約を満たします。

1. **Local Variables:**
   
   * `p_i`: エージェント `i` の割り当て電力。
   * `lambda_i`: エージェント `i` のラグランジュ乗数（シャドープライス）の推定値で、`sum p_i = P_total` 制約に関連します。理想的には、すべての `lambda_i` が同じ最適値 `lambda_star` に収束するべきです。

2. **ADMM Iterations:**
   アルゴリズムはこれらのステップを反復します。
   
   * **Step 1: Primal Update (Local Power Allocation `p_i`)**
     各エージェント `i` は、自身の選好 `p_pref_i` と現在の価格推定値 `lambda_i^k` を考慮して、ローカルな最適化問題を解きます。また、安定性のために、またADMMらしくするために近接項 `(beta/2)*(p_i - p_i^k)^2` を追加します（より単純な形式も存在します）。`beta=0` の場合、これは双対上昇法に近くなります。
     `p_i^{k+1} = argmin_{p_val >= 0} { (p_val - p_pref_i)^2 + lambda_i^k * p_val + (beta/2)*(p_val - p_i^k)^2 }`
     この二次問題の解（導関数をゼロに設定し、非負性を適用することにより）は次のとおりです。
     `p_i^{k+1} = max(0, (2*p_pref_i - lambda_i^k + beta*p_i^k) / (2 + beta))`
   
   * **Step 2: Dual Variable Update (Local Price Update `lambda_i`)**
     このステップには2つのサブステップが含まれます。
     a.  **Estimating Constraint Violation:** 各エージェントは、現在の総電力消費量 `sum_j p_j^{k+1}` の推定値を必要とし、それが `P_total` と一致するかどうかを確認します。この合計は、コンセンサスアルゴリズム（以前使用したDGDのようなものですが、今度は `p_j^{k+1}` を合計するため）を使用して分散的に計算されます。
     b.  **Updating `lambda_i`:** 各エージェントは、古い `lambda_i^k` とグローバルな制約違反の推定値に基づいて `lambda_i` を更新します。一般的な更新は次のとおりです。
     
         `lambda_i^{local_update, k+1} = lambda_i^k + rho * ( (sum_j p_j^{k+1})_i^{estimate} - P_total )`
         ここで、`rho` はペナルティパラメータ（ADMMのステップサイズ）であり、`(sum_j p_j^{k+1})_i^{estimate}` はコンセンサスサブステップから得られたエージェント `i` の合計の推定値です。
   
   * **Step 3: Dual Variable Consensus (Price Agreement `lambda_i`)**
     各 `lambda_i` は理想的には共通の `lambda_star` に収束するべきなので、エージェントは `lambda_i^{local_update, k+1}` の値に対してコンセンサスプロトコルを実行します。
     `lambda_i^{k+1} = Average_Consensus(lambda^{local_update, k+1})`
     (つまり、`lambda_j^{local_update,k+1}` に対してDGDを数回反復実行します)。

この構造は、外側のADMMループと情報集約のための内側のコンセンサスループを含みます。

```matlab
% Distributed Resource Allocation for EV Charging via ADMM
%-----------------------------------------------------------------
clear; clc; close all;

% --- Problem Setup ---
% --- 問題設定 ---
N = 10;                         % Number of EVs (agents) (EVの数（エージェント）)
P_total = 50;                   % Total Power Capacity (kW) (総電力容量 (kW))
p_pref = rand(N,1) * 8 + 3;     % Preferred rates (3 to 11 kW) (優先レート (3から11kW))

fprintf('Preferred charging rates (kW):\n'); disp(p_pref');
fprintf('Sum of preferred rates: %.2f kW\n', sum(p_pref));
fprintf('Total available power: %.2f kW\n\n', P_total);

% --- Communication Graph (Complete Graph for Simplicity) ---
% --- 通信グラフ（簡単のため完全グラフ）---
% W_consensus is for averaging/summing steps within ADMM
% W_consensus はADMM内の平均化/合計ステップ用
W_consensus = ones(N) / N; 
% For DGD-style consensus on lambda, a doubly stochastic matrix is good.
% For simple averaging over a complete graph, 1/N works for each element.
% If graph is not complete, Metropolis-Hastings or similar needed.
% ラムダに関するDGDスタイルのコンセンサスには、二重確率行列が良い。
% 完全グラフ上の単純な平均化の場合、各要素に1/Nが機能する。
% グラフが完全でない場合、Metropolis-Hastingsなどが必要。

% --- ADMM Parameters ---
% --- ADMMパラメータ ---
T_admm = 100;                   % Number of ADMM iterations (outer loop) (ADMM反復回数（外側ループ）)
rho = 0.1;                      % ADMM penalty parameter / dual step size (ADMMペナルティパラメータ/双対ステップサイズ)
beta = 0.05;                    % Proximal term coefficient for p_i update (0 for simpler update) (p_i更新のための近接項係数（0でより単純な更新）)

% --- Consensus Parameters (Inner Loops) ---
% --- コンセンサスパラメータ（内側ループ）---
T_consensus = 20;               % Iterations for inner consensus (sum_p and lambda) (内部コンセンサス（sum_pとラムダ）の反復回数)
alpha_consensus = 0.3;          % Stepsize for DGD within consensus routines (コンセンサスルーチン内のDGDのステップサイズ)

% --- Initialization ---
% --- 初期化 ---
p = p_pref;                     % Initial power allocations (can be p_pref, 0, or P_total/N) (初期電力割り当て（p_pref、0、またはP_total/Nにすることができる）)
% p = ones(N,1) * P_total/N;
lambda = zeros(N,1);            % Initial dual variables (price estimates) (初期双対変数（価格推定値）)

% --- Store History ---
% --- 履歴の保存 ---
Phist_admm = zeros(N, T_admm+1);
Lambdahist_admm = zeros(N, T_admm+1);
Sum_p_hist_admm = zeros(1, T_admm+1);

Phist_admm(:,1) = p;
Lambdahist_admm(:,1) = lambda;
Sum_p_hist_admm(1) = sum(p);

fprintf('Starting ADMM...\n');
% --- ADMM Algorithm ---
% --- ADMMアルゴリズム ---
for k = 1:T_admm
    % --- Step 1: Primal Update (Local p_i solve) ---
    % --- ステップ1：プライマル更新（ローカルp_i解決）---
    % p_i^{k+1} = max(0, (2*p_pref_i - lambda_i^k + beta*p_i^k) / (2 + beta))
    p_prev_iter = p; % Store p_i^k for the proximal term (近接項のためにp_i^kを保存)
    for i = 1:N
        numerator = 2 * p_pref(i) - lambda(i) + beta * p_prev_iter(i);
        denominator = 2 + beta;
        p(i) = max(0, numerator / denominator);
    end

    % --- Step 2a: Distributed Summation of p_i^{k+1} ---
    % --- ステップ2a：p_i^{k+1}の分散合計 ---
    % Each agent needs an estimate of sum(p). We use DGD to average p_i, then multiply by N.
    % Or, DGD to directly sum (more complex DGD variant). Simpler: average then scale.
    % 各エージェントはsum(p)の推定値を必要とする。DGDを使用してp_iを平均化し、その後Nを掛ける。
    % または、DGDで直接合計する（より複雑なDGDバリアント）。より単純：平均化してからスケーリング。

    current_p_for_avg = p; % Values to be averaged (平均化される値)
    avg_p_estimates = current_p_for_avg; % Initial estimate is local value (初期推定値はローカル値)

    % Store history of sum_p estimation for this ADMM iteration (optional, for debug)
    % このADMM反復のsum_p推定の履歴を保存（オプション、デバッグ用）
    % AvgPHistInner = zeros(N, T_consensus+1); AvgPHistInner(:,1) = avg_p_estimates;

    for iter_c = 1:T_consensus
        % DGD for averaging: x_new = W_consensus * x_old - alpha_c * (x_old - values_to_average)
        % Here, we want x_old to converge to mean(values_to_average).
        % The standard DGD "x = Wx - alpha*(x-a)" makes x converge to mean(a).
        % So 'x' is avg_p_estimates, 'a' is current_p_for_avg
        % Using a simpler consensus averaging: new_estimates = W_consensus * old_estimates
        % This simple form requires W_consensus to be doubly stochastic and primitive.
        % For complete graph W_consensus = ones(N)/N, this works.
        % 平均化のためのDGD：x_new = W_consensus * x_old - alpha_c * (x_old - values_to_average)
        % ここでは、x_oldがmean(values_to_average)に収束することを望む。
        % 標準的なDGD "x = Wx - alpha*(x-a)" はxをmean(a)に収束させる。
        % したがって、'x'はavg_p_estimates、'a'はcurrent_p_for_avg
        % より単純なコンセンサス平均化を使用：new_estimates = W_consensus * old_estimates
        % この単純な形式は、W_consensusが二重確率かつ原始的であることを要求する。
        % 完全グラフW_consensus = ones(N)/Nの場合、これは機能する。
        avg_p_estimates = W_consensus * avg_p_estimates; % Iterative averaging (反復平均化)
        % AvgPHistInner(:, iter_c+1) = avg_p_estimates;
    end
    % After T_consensus iterations, avg_p_estimates(i) is agent i's estimate of mean(p)
    % So, agent i's estimate of sum(p) is N * avg_p_estimates(i)
    % T_consensus回の反復後、avg_p_estimates(i)はエージェントiのmean(p)の推定値
    % したがって、エージェントiのsum(p)の推定値は N * avg_p_estimates(i)
    estimated_sum_p_all_agents = N * avg_p_estimates; % Vector of N estimates, ideally all similar (N個の推定値のベクトル、理想的にはすべて類似)

    % For the lambda update, each agent uses its own estimate of the sum.
    % Or, we could pick one (e.g., agent 1's estimate) if we assume good consensus.
    % Let's use each agent's individual estimate for local lambda update.
    % ラムダ更新のために、各エージェントは合計の自身の推定値を使用する。
    % または、良好なコンセンサスを仮定すれば、1つを選ぶこともできる（例：エージェント1の推定値）。
    % ローカルラムダ更新のために、各エージェントの個々の推定値を使用しよう。

    % --- Step 2b: Dual Variable Pre-Update (Local lambda_i update) ---
    % --- ステップ2b：双対変数の事前更新（ローカルlambda_i更新）---
    lambda_local_update = lambda + rho * (estimated_sum_p_all_agents - P_total);

    % --- Step 3: Consensus on Dual Variables (lambda_i) ---
    % --- ステップ3：双対変数（lambda_i）に関するコンセンサス ---
    % Agents run consensus on their lambda_local_update values.
    % エージェントはlambda_local_update値に対してコンセンサスを実行する。
    lambda_for_consensus = lambda_local_update;
    current_lambda_estimates = lambda_for_consensus; 

    % Store history of lambda estimation (optional)
    % ラムダ推定の履歴を保存（オプション）
    % LambdaHistInner = zeros(N, T_consensus+1); LambdaHistInner(:,1) = current_lambda_estimates;

    for iter_c = 1:T_consensus
        % Similar consensus averaging for lambda
        % ラムダのための同様のコンセンサス平均化
        current_lambda_estimates = W_consensus * current_lambda_estimates;
        % LambdaHistInner(:, iter_c+1) = current_lambda_estimates;
    end
    lambda = current_lambda_estimates; % All lambda(i) should be close after consensus (コンセンサス後、すべてのlambda(i)は近いはず)

    % --- Store history for ADMM iteration k ---
    % --- ADMM反復kの履歴を保存 ---
    Phist_admm(:,k+1) = p;
    Lambdahist_admm(:,k+1) = lambda; % Store the consensus-agreed lambda (コンセンサス合意されたラムダを保存)
    Sum_p_hist_admm(k+1) = sum(p); % True sum for plotting global constraint satisfaction (グローバル制約満足度プロット用の真の合計)

    if mod(k, T_admm/10) == 0
        fprintf('ADMM Iter %d/%d: Sum P = %.2f (Target P_total = %.1f), Avg Lambda = %.2f\n', ...
                k, T_admm, sum(p), P_total, mean(lambda));
    end
end
fprintf('ADMM Finished.\n');

% --- Results ---
% --- 結果 ---
p_allocated_admm = Phist_admm(:, T_admm+1);

fprintf('\n--- ADMM Final Allocations ---\n');
disp('EV # | Preferred (kW) | Allocated (kW) | Final Lambda_i');
for i=1:N
    fprintf('%4d | %14.2f | %14.2f | %14.4f\n', i, p_pref(i), p_allocated_admm(i), lambda(i));
end
fprintf('--------------------------------------------------------------------\n');
fprintf('Sum of Preferred Rates: %.2f kW\n', sum(p_pref));
fprintf('Sum of Allocated Rates (ADMM): %.2f kW (Target: %.2f kW)\n', sum(p_allocated_admm), P_total);
fprintf('Final Lambda values std dev: %.4e (should be small if consensus worked)\n', std(lambda));


% --- Analytical Optimal Solution (for comparison) ---
% --- 解析的な最適解（比較用）---
% Solved via KKT conditions:
% KKT条件を介して解決：
% L = sum_i (p_i - p_pref_i)^2 - nu (sum_i p_i - P_total) - sum_i mu_i p_i
% dL/dp_i = 2(p_i - p_pref_i) - nu - mu_i = 0
% If p_i > 0, then mu_i = 0 => p_i = p_pref_i + nu/2
% If p_i = 0, then mu_i >= 0 => 2(-p_pref_i) - nu - mu_i = 0 => nu <= -2*p_pref_i
% The single 'nu' here corresponds to '-lambda_star' from our ADMM.
% So, p_i_opt = p_pref_i - lambda_star/2
% We need to find lambda_star such that sum(max(0, p_pref_i - lambda_star/2)) = P_total
% This can be found by a 1D root search for lambda_star.
% ここでの単一の'nu'は、ADMMの'-lambda_star'に対応する。
% したがって、p_i_opt = p_pref_i - lambda_star/2
% sum(max(0, p_pref_i - lambda_star/2)) = P_total となるようなlambda_starを見つける必要がある。
% これはlambda_starの1次元根探索で見つけることができる。
f_to_solve = @(l_star) sum(max(0, p_pref - l_star/2)) - P_total;
% Sensible search range for lambda_star:
% If all p_pref are high, lambda_star will be positive (to reduce allocations)
% If all p_pref are low, lambda_star will be negative (to increase allocations)
% lambda_starの適切な探索範囲：
% すべてのp_prefが高い場合、lambda_starは正になる（割り当てを減らすため）
% すべてのp_prefが低い場合、lambda_starは負になる（割り当てを増やすため）
min_l = 2 * (min(p_pref) - max(p_pref) - P_total/N); % Heuristic lower bound (ヒューリスティックな下限)
max_l = 2 * (max(p_pref) - min(p_pref) + P_total/N); % Heuristic upper bound (ヒューリスティックな上限)
try
    lambda_star_optimal = fzero(f_to_solve, mean(lambda)); % Start search near ADMM's result (ADMMの結果の近くで探索を開始)
catch
    try 
        lambda_star_optimal = fzero(f_to_solve, 0); % Try starting at 0 (0から開始してみる)
    catch
        fprintf('fzero failed to find analytical lambda_star. Plotting without it.\n');
        lambda_star_optimal = NaN;
    end
end
p_optimal_analytical = max(0, p_pref - lambda_star_optimal/2);


% --- Plotting ---
% --- プロット ---
% 1. Convergence of allocations p_i
% 1. 割り当てp_iの収束
figure;
plot(0:T_admm, Phist_admm', 'LineWidth', 1.2);
xlabel('ADMM Iteration k');
ylabel('Allocated Power p_i^{(k)} (kW)');
title(sprintf('ADMM: EV Power Allocations (N=%d, rho=%.2f, beta=%.2f)', N, rho, beta));
legendStrings_p = arrayfun(@(i) sprintf('EV %d',i), 1:N, 'UniformOutput', false);
legend(legendStrings_p,'Location','best');
grid on;

% 2. Convergence of dual variables lambda_i
% 2. 双対変数lambda_iの収束
figure;
plot(0:T_admm, Lambdahist_admm', 'LineWidth', 1.2);
hold on;
if ~isnan(lambda_star_optimal)
    yline(lambda_star_optimal, 'k--', 'LineWidth', 2, 'Label', 'Optimal \lambda* (analytical)');
end
xlabel('ADMM Iteration k');
ylabel('Dual Variable \lambda_i^{(k)}');
title(sprintf('ADMM: Convergence of Dual Variables (Price Estimates)'));
legendStrings_l = arrayfun(@(i) sprintf('\\lambda_%d',i), 1:N, 'UniformOutput', false);
if ~isnan(lambda_star_optimal)
    legend([legendStrings_l, {'Optimal \lambda*'}],'Location','best');
else
    legend(legendStrings_l,'Location','best');
end
grid on;

% 3. Convergence of Sum p_i to P_total
% 3. Sum p_i の P_total への収束
figure;
plot(0:T_admm, Sum_p_hist_admm, 'b-', 'LineWidth', 1.5);
hold on;
yline(P_total, 'r--', 'LineWidth', 2);
xlabel('ADMM Iteration k');
ylabel('Total Allocated Power \Sigma p_i^{(k)} (kW)');
title('ADMM: Convergence of Total Power to P_{total}');
legend('Sum p_i^{(k)}', 'P_{total}');
grid on; ylim([P_total*0.8, P_total*1.2]); % Adjust ylim if needed (必要に応じてylimを調整)

% 4. Final Bar Chart Comparison
% 4. 最終的な棒グラフ比較
figure;
bar_data = [p_pref, p_allocated_admm];
legend_items = {'Preferred Rate', 'ADMM Allocated Rate'};
if ~isnan(lambda_star_optimal)
    bar_data = [bar_data, p_optimal_analytical];
    legend_items = [legend_items, {'Analytical Optimal'}];
end

b = bar(1:N, bar_data, 'grouped');
xlabel('Electric Vehicle (Agent)');
ylabel('Charging Rate (kW)');
title('Preferred vs. Allocated Charging Rates (ADMM)');
xticks(1:N);
legend(legend_items, 'Location', 'northwest');
grid on;

if sum(p_pref) > P_total
    annotation_text = sprintf('Demand (%.1fkW) > Supply (%.1fkW)\nRates adjusted by ADMM.', sum(p_pref), P_total);
    annotation_color = 'yellow';
elseif sum(p_pref) < P_total
    annotation_text = sprintf('Demand (%.1fkW) < Supply (%.1fkW)\nRates adjusted by ADMM.', sum(p_pref), P_total);
    annotation_color = 'cyan';
else
    annotation_text = sprintf('Demand (%.1fkW) == Supply (%.1fkW)', sum(p_pref), P_total);
    annotation_color = 'green';
end
annotation('textbox', [0.55, 0.15, 0.1, 0.1], 'String', annotation_text, ...
        'FitBoxToText','on', 'BackgroundColor', annotation_color, 'FontSize', 8);
```

**Key Elements and Changes:**

1. **ADMM Framework:** プライマル更新、双対更新、およびコンセンサスという中核的な反復構造は、分散問題に適用されるADMMの特徴です。
2. **Dual Variables (`lambda_i`):** これらは電力の「価格」を表します。ADMMは、エージェントが市場をクリアする（つまり、需要を `P_total` に一致させる）最適な価格に収束するのを助けます。
3. **Proximal Term (`beta`):** `p_i` 更新における `(beta/2)*(p_val - p_i^k)^2` 項は近接項です。これは更新を正則化し、`p_i` が1回の反復で急激に変化するのを防ぎます。これにより、特に `rho` がうまく調整されていない場合に、安定性と収束性が向上します。`beta=0` の場合、`p_i` 更新はより単純で、（拡張）ラグランジアン法または双対上昇法のステップに対応します。
4. **Penalty Parameter (`rho`):** これはADMMにおいて重要です。目的関数の最小化と制約充足のバランスを取ります。`rho`（および `beta`）の調整は、収束速度に大きな影響を与える可能性があります。
5. **Nested Consensus:**
   * **Summation of `p_i`:** `sum p_i = P_total` 制約を確認するために、エージェントは現在の合計を知る必要があります。これは、分散平均化（または合計）プロトコル（ここでは `W_consensus` を使用した単純化された反復平均化）を実行することによって達成されます。
   * **Consensus on `lambda_i`:** 各エージェントが（推定された）グローバルな電力不一致に基づいて「望ましい」次の `lambda_i` を計算した後、すべての `lambda_i` が共通の値に収束するように別のコンセンサスラウンドを実行します。これにより、ネットワーク全体で価格シグナルが一貫性を持ちます。
6. **Modularity:** コンセンサスステップはサブルーチンと見なすことができます。より高度なコンセンサスアルゴリズム（例：特定のグラフトポロジーに対するより速い収束率を持つもの、または最適化されたステップサイズのDGD）をプラグインすることができます。

**How it Compares to the Simpler DGD for Averaging:**

* 以前のDGDの例は `mean(a)` を計算するために使用され、その後、最適な `p_i` の解析式に*代入*されました。
* このADMMアプローチは、制約付き最適化問題を反復的に*直接解き*ます。DGDのようなコンセンサスは、ADMMの*内部ループ*コンポーネントとなり、情報交換（`p` の合計と `lambda` の平均化）に使用されます。
* ADMMは一般に、制約の処理に対してよりロバストであり、より弱い条件下で、または平均化後の単純な解析解が利用できないより複雑な問題に対して収束することができます。

このADMMの例は、結合制約を持つ分散最適化問題に取り組むための、より洗練された一般化可能な方法を提供します。

### 結果

```matlab
Starting ADMM...
ADMM Iter 10/100: Sum P = 50.01 (Target P_total = 50.0), Avg Lambda = 1.03
ADMM Iter 20/100: Sum P = 50.00 (Target P_total = 50.0), Avg Lambda = 1.03
ADMM Iter 30/100: Sum P = 50.00 (Target P_total = 50.0), Avg Lambda = 1.03
ADMM Iter 40/100: Sum P = 50.00 (Target P_total = 50.0), Avg Lambda = 1.03
ADMM Iter 50/100: Sum P = 50.00 (Target P_total = 50.0), Avg Lambda = 1.03
ADMM Iter 60/100: Sum P = 50.00 (Target P_total = 50.0), Avg Lambda = 1.03
ADMM Iter 70/100: Sum P = 50.00 (Target P_total = 50.0), Avg Lambda = 1.03
ADMM Iter 80/100: Sum P = 50.00 (Target P_total = 50.0), Avg Lambda = 1.03
ADMM Iter 90/100: Sum P = 50.00 (Target P_total = 50.0), Avg Lambda = 1.03
ADMM Iter 100/100: Sum P = 50.00 (Target P_total = 50.0), Avg Lambda = 1.03
ADMM Finished.

ADMM Final Allocations ---
EV # | Preferred (kW) | Allocated (kW) | Final Lambda_i
   1 |           6.34 |           5.82 |         1.0341
   2 |           8.76 |           8.25 |         1.0341
   3 |           3.00 |           2.48 |         1.0341
   4 |           5.42 |           4.90 |         1.0341
   5 |           4.17 |           3.66 |         1.0341
   6 |           3.74 |           3.22 |         1.0341
   7 |           4.49 |           3.97 |         1.0341
   8 |           5.76 |           5.25 |         1.0341
   9 |           6.17 |           5.66 |         1.0341
  10 |           7.31 |           6.79 |         1.0341
--------------------------------------------------------------------
Sum of Preferred Rates: 55.17 kW
Sum of Allocated Rates (ADMM): 50.00 kW (Target: 50.00 kW)
Final Lambda values std dev: 0.0000e+00 (should be small if consensus worked)
```

![](admm_1.png)

![](admm_2.png)

![](admm_3.png)

![](admm_4.png)

---

次に、MATLABのGlobal Optimization Toolboxの `particleswarm` 関数を使用して、EV充電リソース割り当て問題を粒子群最適化 (PSO) を用いて解いてみましょう。

## Solve with Particle Swarm Optimization (PSO)

**Challenge with PSO for this Problem:**
標準の `particleswarm` 関数は、境界制約 (`lb`, `ub`) を直接扱いますが、`sum(p_i) = P_total` のような一般的な線形または非線形の等式制約は扱いません。これは**ペナルティ法**を使用して対処します。合計制約に違反する解に大きなペナルティを課すように目的関数を変更します。

**Modified Optimization Problem for PSO:**

Minimize `f(p) = sum_i (p_i - p_pref_i)^2 + C * (sum_i p_i - P_total)^2`
Subject to:

1. `0 <= p_i <= P_total` (または、既知であればより厳しい上限、例：個々のEVの最大充電レート)

ここで `C` は大きなペナルティ係数です。

```matlab
% Distributed Resource Allocation for EV Charging via PSO
%-----------------------------------------------------------------
clear; clc; close all;

% --- Problem Setup ---
% --- 問題設定 ---
N = 10;                         % Number of EVs (agents) (EVの数（エージェント）)
P_total = 50;                   % Total Power Capacity (kW) (総電力容量 (kW))
p_pref = rand(N,1) * 8 + 3;     % Preferred rates (3 to 11 kW) (優先レート (3から11kW))

fprintf('Preferred charging rates (kW):\n'); disp(p_pref');
fprintf('Sum of preferred rates: %.2f kW\n', sum(p_pref));
fprintf('Total available power: %.2f kW\n\n', P_total);

% --- PSO Parameters ---
% --- PSOパラメータ ---
nvars = N;                      % Number of optimization variables (power for each EV) (最適化変数の数（各EVの電力）)

% Lower bounds (p_i >= 0)
% 下限 (p_i >= 0)
lb = zeros(N,1);

% Upper bounds (0 <= p_i <= P_total theoretically, or max individual charge rate)
% For simplicity, let's use P_total as a loose upper bound for each.
% A tighter bound could be max(p_pref) or a physical EV charging limit.
% 上限 (理論的には 0 <= p_i <= P_total、または個々の最大充電レート)
% 簡単のため、各EVの緩い上限としてP_totalを使用する。
% より厳しい境界はmax(p_pref)や物理的なEV充電制限など。
ub = P_total * ones(N,1);
% ub = min(P_total, max(15, p_pref * 1.5)); % Example of slightly tighter UB (少し厳しい上限の例)

% Penalty coefficient for the sum constraint
% 合計制約のペナルティ係数
penalty_coefficient = 1000; % This might need tuning (調整が必要な場合がある)

% --- Objective Function for PSO (with penalty) ---
% --- PSOの目的関数（ペナルティ付き）---
% The input 'p' to this function will be a row vector (PSO standard)
% この関数への入力'p'は行ベクトルになる（PSO標準）
objectiveFunction = @(p) sum((p' - p_pref).^2) + ...
                         penalty_coefficient * (sum(p) - P_total)^2;
% Note: p' is used because p_pref is a column vector.
% PSO passes 'p' as a row vector to the objective function.
% 注意：p_prefは列ベクトルなのでp'を使用。
% PSOは'p'を行ベクトルとして目的関数に渡す。

% --- PSO Options ---
% --- PSOオプション ---
options = optimoptions('particleswarm', ...
    'SwarmSize', max(50, 10*nvars), ... % Number of particles (粒子数)
    'MaxIterations', 200*nvars, ...    % Max iterations (最大反復回数)
    'Display', 'iter', ...           % Show iteration progress (反復の進行状況を表示)
    'ObjectiveLimit', 1e-4, ...      % Stop if objective is good enough (目的が十分に良ければ停止)
    'FunctionTolerance', 1e-6, ...   % Stop if objective change is small (目的の変化が小さければ停止)
    'UseParallel', false);            % Set to true if you have Parallel Computing Toolbox
                                      % and problem is large enough.
                                      % Parallel Computing Toolboxがあり、問題が十分に大きい場合はtrueに設定。

fprintf('Starting PSO...\n');
% --- Run PSO ---
% --- PSO実行 ---
% particleswarm(objectiveFunction, nvars, lb, ub, options)
[p_optimal_pso_raw, fval_pso, exitflag, output_pso] = particleswarm(objectiveFunction, nvars, lb, ub, options);

fprintf('PSO Finished.\n');
fprintf('PSO Exit Flag: %d\n', exitflag);
disp(output_pso.message);
fprintf('Objective function value at solution: %.4e\n', fval_pso);
fprintf('Constraint term value: %.4e * (%.4f - %.1f)^2 = %.4e\n', ...
    penalty_coefficient, sum(p_optimal_pso_raw), P_total, ...
    penalty_coefficient * (sum(p_optimal_pso_raw) - P_total)^2);

% --- Post-processing the PSO solution ---
% --- PSO解の後処理 ---
% PSO solution is a row vector, convert to column for consistency
% PSO解は行ベクトルなので、一貫性のために列ベクトルに変換
p_allocated_pso = p_optimal_pso_raw';

% Check the sum constraint satisfaction
% 合計制約の満足度をチェック
sum_p_pso = sum(p_allocated_pso);
fprintf('Sum of allocated rates (PSO raw): %.4f kW (Target: %.2f kW)\n', sum_p_pso, P_total);

% If the sum is not exactly P_total due to penalty approximation,
% we can apply a simple scaling as a heuristic, provided the solution is "close".
% This is common practice if the penalty method gets us near the feasible region.
% ペナルティ近似のために合計が正確にP_totalでない場合、
% 解が「近い」場合に限り、ヒューリスティックとして単純なスケーリングを適用できる。
% これは、ペナルティ法が実行可能領域の近くに到達した場合の一般的な慣行。
if abs(sum_p_pso - P_total) > 1e-3 && sum_p_pso > 1e-6 % Add tolerance and check if sum is not zero (許容誤差を追加し、合計がゼロでないことを確認)
    fprintf('Normalizing PSO solution to meet sum constraint exactly.\n');
    p_allocated_pso_normalized = p_allocated_pso * (P_total / sum_p_pso);
    % Ensure non-negativity again after scaling (though unlikely to be an issue if lb was 0 and scaling factor > 0)
    % スケーリング後に再度非負性を保証（lbが0でスケーリング係数が>0であれば問題になる可能性は低い）
    p_allocated_pso_normalized = max(0, p_allocated_pso_normalized);
    % If this re-normalization drastically changes the solution or violates other complex constraints
    % (not present here), then the penalty method or PSO tuning needs more work.
    % この再正規化が解を大幅に変更したり、他の複雑な制約（ここでは存在しない）に違反したりする場合、
    % ペナルティ法またはPSOの調整がさらに必要。
else
    p_allocated_pso_normalized = p_allocated_pso;
end
% Re-check sum after potential normalization
% 正規化の可能性の後に合計を再確認
sum_p_pso_normalized = sum(p_allocated_pso_normalized);

fprintf('\n--- PSO Final Allocations (after potential normalization) ---\n');
disp('EV # | Preferred (kW) | Allocated (kW)');
for i=1:N
    fprintf('%4d | %14.2f | %14.2f\n', i, p_pref(i), p_allocated_pso_normalized(i));
end
fprintf('--------------------------------------------------------------------\n');
fprintf('Sum of Preferred Rates: %.2f kW\n', sum(p_pref));
fprintf('Sum of Allocated Rates (PSO normalized): %.2f kW (Target: %.2f kW)\n', sum_p_pso_normalized, P_total);
fprintf('Original objective value (without penalty) for PSO solution: %.4f\n', sum((p_allocated_pso_normalized - p_pref).^2));


% --- Analytical Optimal Solution (for comparison, same as in ADMM example) ---
% --- 解析的な最適解（比較用、ADMMの例と同じ）---
f_to_solve = @(lambda_star) sum(max(0, p_pref - lambda_star/2)) - P_total;
try
    lambda_star_optimal = fzero(f_to_solve, 0); 
    p_optimal_analytical = max(0, p_pref - lambda_star_optimal/2);
    analytical_available = true;
    fprintf('Analytical optimal objective value: %.4f\n', sum((p_optimal_analytical - p_pref).^2));
catch
    fprintf('fzero failed to find analytical lambda_star. Plotting without it.\n');
    p_optimal_analytical = nan(N,1);
    analytical_available = false;
end

% --- Plotting ---
% --- プロット ---
figure;
bar_data = [p_pref, p_allocated_pso_normalized];
legend_items = {'Preferred Rate', 'PSO Allocated Rate'};
if analytical_available
    bar_data = [bar_data, p_optimal_analytical];
    legend_items = [legend_items, {'Analytical Optimal'}];
end

b = bar(1:N, bar_data, 'grouped');
xlabel('Electric Vehicle (Agent)');
ylabel('Charging Rate (kW)');
title(sprintf('Preferred vs. Allocated Charging Rates (PSO, Penalty Coeff: %g)', penalty_coefficient));
xticks(1:N);
legend(legend_items, 'Location', 'northwest');
grid on;

if sum(p_pref) > P_total
    annotation_text = sprintf('Demand (%.1fkW) > Supply (%.1fkW)\nRates adjusted by PSO.', sum(p_pref), P_total);
    annotation_color = 'yellow';
elseif sum(p_pref) < P_total
    annotation_text = sprintf('Demand (%.1fkW) < Supply (%.1fkW)\nRates adjusted by PSO.', sum(p_pref), P_total);
    annotation_color = 'cyan';
else
    annotation_text = sprintf('Demand (%.1fkW) == Supply (%.1fkW)', sum(p_pref), P_total);
    annotation_color = 'green';
end
annotation('textbox', [0.55, 0.15, 0.1, 0.1], 'String', annotation_text, ...
        'FitBoxToText','on', 'BackgroundColor', annotation_color, 'FontSize', 8);

% --- Convergence Plot (if you want to see PSO progress, requires custom output function) ---
% --- 収束プロット（PSOの進行状況を確認したい場合、カスタム出力関数が必要）---
% To get iteration-wise best fitness for plotting convergence:
% 収束プロットのために反復ごとの最良適合度を得るには：
% 1. Define an output function:
% 1. 出力関数を定義：
%    function [state,options,optchanged] = psoIterDisplay(options,state,flag)
%        optchanged = false;
%        if strcmp(flag,'iter')
%            if ~isfield(state,'BestFitnessHistory')
%                state.BestFitnessHistory = [];
%            end
%            state.BestFitnessHistory = [state.BestFitnessHistory, state.BestFitness];
%        end
%    end
% 2. Add 'OutputFcn', @psoIterDisplay to optimoptions
% 2. optimoptionsに'OutputFcn', @psoIterDisplayを追加
% 3. After running PSO:
% 3. PSO実行後：
%    figure;
%    plot(output_pso.BestFitnessHistory); % Assuming BestFitnessHistory was populated (BestFitnessHistoryが入力されたと仮定)
%    xlabel('Iteration'); ylabel('Best Objective Value (incl. penalty)');
%    title('PSO Convergence'); grid on;
% Note: The standard 'Display','iter' option gives some info, but for a plot,
% a custom output function or capturing iteration data is needed.
% The 'output_pso' struct doesn't store the full history by default.
% Let's try to use a built-in plot function if available or make a simple one.
% 注意：標準の'Display','iter'オプションはいくつかの情報を提供するが、プロットのためには、
% カスタム出力関数または反復データのキャプチャが必要。
% 'output_pso'構造体はデフォルトで完全な履歴を保存しない。
% 利用可能であれば組み込みプロット関数を使用するか、単純なものを作成してみよう。
if isfield(options, 'PlotFcn') && ~isempty(options.PlotFcn)
    % If a plot function was specified and ran, figures might already exist.
    % For example, 'PlotFcns', @psoplotswarm shows particle movement.
    % @psoplotbestf shows best fitness.
    % プロット関数が指定されて実行された場合、図が既に存在する可能性がある。
    % 例：'PlotFcns', @psoplotswarm は粒子の動きを示す。
    % @psoplotbestf は最良適合度を示す。
    fprintf("If 'PlotFcns' was used in options, relevant plots might already be displayed.\n")
else
    % If no PlotFcn was used, we can try to plot the objective value from fval_pso
    % (which is only the final value). To get a convergence plot, you'd typically
    % need to set options.PlotFcns = @psoplotbestf;
    % PlotFcnが使用されなかった場合、fval_psoから目的値をプロットしてみることができる
    % （これは最終値のみ）。収束プロットを得るには、通常
    % options.PlotFcns = @psoplotbestf; を設定する必要がある。
    fprintf("To see PSO convergence plot, set options.PlotFcns = @psoplotbestf;\n");
    fprintf("Example: options = optimoptions(options, 'PlotFcns', @psoplotbestf);\n");
end
```

**Explanation of PSO Implementation:**

1. **Problem Formulation for PSO:**
   
   * 中核となるアイデアは、等式制約 `sum(p_i) = P_total` を処理するためにペナルティ関数を使用することです。
   * 目的関数は次のようになります: `f(p) = sum_i (p_i - p_pref_i)^2 + penalty_coefficient * (sum_i p_i - P_total)^2`。
   * `penalty_coefficient` は、PSOが `sum(p_i)` が `P_total` に非常に近い解を見つけるように強制するのに十分な大きさでなければなりません。その値は調整が必要な場合があります。
   * 制約 `p_i >= 0` および `p_i <= P_total` (またはより厳しい個々の上限) は、`particleswarm` の `lb` (下限) および `ub` (上限) 引数によって直接処理されます。

2. **`particleswarm` Function Call:**
   
   * `objectiveFunction`: (ペナルティを含む) フィットネス関数へのハンドル。
   * `nvars`: 変数の数 (`N`、EVの数)。
   * `lb`: 下限のベクトル (すべてゼロ)。
   * `ub`: 上限のベクトル (例: 各EVに対して `P_total`、または個々の最大充電レート)。
   * `options`: PSOの動作を制御する構造体:
     * `SwarmSize`: スウォーム内の粒子数。
     * `MaxIterations`: 最大反復回数。
     * `Display`: `'iter'` はコマンドウィンドウに進捗を表示します。
     * `ObjectiveLimit`, `FunctionTolerance`: 停止基準。
     * `UseParallel`: Parallel Computing Toolbox がある場合、計算を高速化できます。
     * 実行中にこれまでに検出された最良の目的値のライブプロットを表示するために、オプションに `'PlotFcns', @psoplotbestf` を追加することもできます。

3. **Post-Processing:**
   
   * `particleswarm` からの解 `p_optimal_pso_raw` は、電力割り当てのセットです。
   * **Constraint Check:** ペナルティ法が制約をどれだけうまく強制したかを確認するために、`sum(p_optimal_pso_raw)` を明示的にチェックします。
   * **Normalization (Heuristic):** 合計が `P_total` からわずかにずれている (しかし近い) 場合、一般的なヒューリスティックは解を比例的にスケーリングすることです: `p_allocated_pso_normalized = p_allocated_pso * (P_total / sum_p_pso)`。これにより、合計制約が正確に満たされるようになります。その後、スケーリングによって値がわずかに負になった場合に備えて ( `sum_p_pso` と `P_total` が正で近い場合はありそうにないですが)、`max(0, ...)` を再適用します。このステップは、PSOによって見つけられた電力の*分布*が良いと仮定していますが、ペナルティ近似のために合計が完全に正確ではない場合でも同様です。

4. **Comparison:**
   
   * 結果は、優先レートおよび解析解 (計算可能な場合) と棒グラフで比較されます。

**Advantages of PSO for this type of problem:**

* **Derivative-Free:** 勾配情報を必要としないため、複雑なまたは微分不可能な目的関数に適しています (ただし、ここでの目的関数は微分可能です)。
* **Global Search:** 一部の勾配ベースの方法と比較して、特に探索空間が複雑な場合に、局所最適解に陥る可能性が低くなります。
* **Ease of Handling Bounds:** `lb` と `ub` は自然に処理されます。

**Disadvantages/Considerations:**

* **Equality Constraints:** ペナルティ関数を介した等式制約の処理は厄介な場合があります。ペナルティ係数は慎重な調整が必要です。低すぎると制約に違反し、高すぎると探索ランドスケープを歪め、元の問題の真の最適解を見つけるのが難しくなる可能性があります。
* **Computational Cost:** PSOは、そのようなソルバーが適用可能な問題に対して、特化した凸最適化ソルバー (適切なアルゴリズムを持つ `fmincon` など) よりも計算コストが高くなる可能性があります。
* **No Guarantee of Optimality:** PSOのようなメタヒューリスティクスは良い解を提供しますが、通常、有限時間でグローバルな最適解を見つけることを保証しません (特に連続問題の場合)。
* **Stochastic Nature:** PSOを複数回実行すると、わずかに異なる結果が得られる場合があります。

このPSOアプローチは、特に問題構造がより複雑な場合や、勾配情報が利用できないか信頼できない場合に、リソース割り当て問題を解決するための実行可能な代替手段を提供します。

### 結果

```matlab
Starting PSO...
Optimization ended: relative change in the objective value 
over the last OPTIONS.MaxStallIterations iterations is less than OPTIONS.FunctionTolerance.
PSO Finished.
PSO Exit Flag: 1
Optimization ended: relative change in the objective value 
over the last OPTIONS.MaxStallIterations iterations is less than OPTIONS.FunctionTolerance.
Objective function value at solution: 3.6154e+01
Constraint term value: 1.0000e+03 * (49.9936 - 50.0)^2 = 4.0709e-02
Sum of allocated rates (PSO raw): 49.9936 kW (Target: 50.00 kW)
Normalizing PSO solution to meet sum constraint exactly.

--- PSO Final Allocations (after potential normalization) ---
EV # | Preferred (kW) | Allocated (kW)
   1 |           6.34 |           5.87
   2 |           8.76 |          10.24
   3 |           3.00 |           4.60
   4 |           5.42 |           0.30
   5 |           4.17 |           2.35
   6 |           3.74 |           4.54
   7 |           4.49 |           4.28
   8 |           5.76 |           5.51
   9 |           6.17 |           5.30
  10 |           7.31 |           7.00
--------------------------------------------------------------------
Sum of Preferred Rates: 55.17 kW
Sum of Allocated Rates (PSO normalized): 50.00 kW (Target: 50.00 kW)
Original objective value (without penalty) for PSO solution: 36.1156
Analytical optimal objective value: 2.6732
```

![](pso.png)

---

## Solve with GA

遺伝的アルゴリズム（GA）は、MATLAB の Global Optimization Toolbox に含まれる集団ベースのメタヒューリスティック手法です。自然選択のプロセスに着想を得ており、選択、交叉、突然変異といったオペレータを用いて解の集団を進化させ、最適解に近づけます。本問題における `ga` の標準的な `particleswarm` に対する大きな利点は、線形等式制約を含む各種制約を洗練された方法で直接扱える点にあります。
**EV 充電問題への GA 適用：** 

- **目的関数** `sum_i (p_i - p_pref_i)^2` を最小化

- **制約** `0 <= p_i <= P_total`（またはより厳しい個別上限）

— 境界制約として `ga` が直接扱います。`sum_i p_i = P_total`

— 線形等式制約で、`Aeq`／`beq` 引数を使い、候補解を可行部分空間に射影する形で扱います。

このように等式制約を直接組み込むことで、PSO のペナルティ法に頼る場合と比べ、より高精度かつ可行な解が得られやすくなります。

```matlab
% 遺伝的アルゴリズム（GA）による EV 充電リソース割り当て
%-----------------------------------------------------------------
clear; clc; close all;

% --- 問題設定 ---
N = 10;                         % EV（エージェント）数
P_total = 50;                   % 総電力容量 (kW)
% 比較のため、p_pref を乱数生成する場合はシードを固定
rng(42);                        % 任意の固定シードでも可
p_pref = rand(N,1) * 8 + 3;     % 希望充電レート (3～11 kW)（列ベクトル）

fprintf('希望充電レート (kW):\n'); disp(p_pref');
fprintf('希望レート合計: %.2f kW\n', sum(p_pref));
fprintf('総利用可能電力: %.2f kW\n\n', P_total);

% --- GA パラメータ ---
nvars = N;                      % 最適化変数数（各 EV の充電出力）

% 目的関数：GA は 'x' を行ベクトルとして渡す
% p_pref は列ベクトル
objectiveFunction = @(p_row) sum((p_row' - p_pref).^2);

% 境界制約
lb = zeros(1, nvars);           % 下限 (p_i >= 0)、GA は行ベクトルを期待
% 上限 (0 <= p_i <= P_total、または個別最大充電レート)
ub = P_total * ones(1, nvars);  % GA は行ベクトルを期待
% 例：より厳しい上限の例（行ベクトルを保証）
% ub = min(P_total, max(15, p_pref' * 1.5));

% 線形等式制約: sum(p_i) = P_total
Aeq = ones(1, nvars);           % p1 + p2 + … + pN の係数
beq = P_total;                  % 合計が P_total に等しいべき

% 線形不等式制約なし
A = [];
b = [];

% 非線形制約関数なし
nonlcon = [];

% 整数制約なし
IntCon = [];

% --- GA オプション ---
options = optimoptions('ga', ...
    'PopulationSize',      max(50,  10*nvars), ...  % 集団の個体数
    'MaxGenerations',      200 * nvars/10, ...      % 最大世代数
    'Display',             'iter', ...              % 進捗を表示
    'PlotFcn',             {@gaplotbestf, @gaplotstopping}, ... % 最良適合度＆停止条件プロット
    'FunctionTolerance',   1e-7, ...                % 目的関数変化小 → 停止
    'ConstraintTolerance', 1e-7, ...                % 制約満足の許容誤差
    'UseParallel',         false);                  % 並列計算する場合は true

fprintf('GA 開始...\n');
% --- GA 実行 ---
[p_optimal_ga_row, fval_ga, exitflag_ga, output_ga, population_ga, scores_ga] = ...
    ga(objectiveFunction, nvars, A, b, Aeq, beq, lb, ub, nonlcon, IntCon, options);

fprintf('GA 終了。\n');
fprintf('GA Exit Flag: %d\n', exitflag_ga);
disp(output_ga.message);
fprintf('得られた解の目的関数値: %.4e\n', fval_ga);

% --- GA 解の後処理 ---
% GA の解は行ベクトルなので，整合性のため列ベクトルに変換
p_allocated_ga = p_optimal_ga_row';

% 総和制約の満足度を確認（高精度であるはず）
sum_p_ga = sum(p_allocated_ga);
fprintf('GA 割当てレート合計: %.4f kW (目標: %.2f kW)\n', sum_p_ga, P_total);
fprintf('制約からのずれ: %.4e\n', sum_p_ga - P_total);

fprintf('\n--- GA 最終割当 ---\n');
disp('EV # | 希望 (kW) | 割当 (kW)');
for i=1:N
    fprintf('%4d | %10.2f | %10.2f\n', i, p_pref(i), p_allocated_ga(i));
end
fprintf('--------------------------------------------------------------------\n');
fprintf('希望レート合計: %.2f kW\n', sum(p_pref));
fprintf('GA 割当て合計: %.2f kW (目標: %.2f kW)\n', sum(p_allocated_ga), P_total);
fprintf('GA 解の目的値 (二乗誤差和): %.4f\n', fval_ga);

% --- 解析的最適解（ADMM/PSO例と同様に比較用） ---
% KKT 条件により解く: p_i_opt = max(0, p_pref_i - lambda_star/2)
% sum(max(0, p_pref_i - lambda_star/2)) = P_total となる lambda_star を求める
f_to_solve = @(lambda_s) sum(max(0, p_pref - lambda_s/2)) - P_total;
try
    lambda_star_optimal = fzero(f_to_solve, [-100, 100]); % 適当な範囲で探索
    if isnan(lambda_star_optimal) || abs(f_to_solve(lambda_star_optimal))>1e-3
        lambda_star_optimal = fzero(f_to_solve, 0);       % 別の初期値を試す
    end
    p_optimal_analytical = max(0, p_pref - lambda_star_optimal/2);
    analytical_available = true;
    fprintf('解析的最適解の目的値: %.4f\n', sum((p_optimal_analytical - p_pref).^2));
    fprintf('解析的解の合計: %.4f\n', sum(p_optimal_analytical));
catch ME
    fprintf('解析的 lambda_star の探索に失敗: %s\n', ME.message);
    p_optimal_analytical = nan(N,1);
    analytical_available = false;
end

% --- プロット ---
figure;  % 棒グラフ用の新規図
bar_data = [p_pref, p_allocated_ga];
legend_items = {'希望レート', 'GA 割当てレート'};
if analytical_available && ~any(isnan(p_optimal_analytical))
    bar_data = [bar_data, p_optimal_analytical];
    legend_items = [legend_items, {'解析的最適解'}];
end

b = bar(1:N, bar_data, 'grouped');
xlabel('Electric Vehicle (Agent)');
ylabel('Charging Rate (kW)');
title(sprintf('希望 vs. GA 割当てレート (GA, N=%d)', N));
xticks(1:N);
legend(legend_items, 'Location','northwest');
grid on;

if sum(p_pref) > P_total
    annotation_text = sprintf('需要 (%.1fkW) > 供給 (%.1fkW)\nGA で調整済み。', sum(p_pref), P_total);
    annotation_color = 'yellow';
elseif sum(p_pref) < P_total
    annotation_text = sprintf('需要 (%.1fkW) < 供給 (%.1fkW)\nGA で調整済み。', sum(p_pref), P_total);
    annotation_color = 'cyan';
else
    annotation_text = sprintf('需要 (%.1fkW) == 供給 (%.1fkW)', sum(p_pref), P_total);
    annotation_color = 'green';
end
annotation('textbox', [0.55,0.15,0.1,0.1], 'String', annotation_text, ...
           'FitBoxToText','on', 'BackgroundColor',annotation_color, 'FontSize',8);

% GA のプロット（最良適合度と停止条件）は PlotFcn によって別図で生成されます。
disp('最良適合度と停止条件のプロットが別図として表示されているはずです。');
```

### 結果

```matlab
ga stopped because it exceeded options.MaxGenerations.
GA Finished.
GA Exit Flag: 0
ga stopped because it exceeded options.MaxGenerations.
Objective function value at solution: 2.6765e+00
Sum of allocated rates (GA): 50.0000 kW (Target: 50.00 kW)
Deviation from sum constraint: 6.3949e-14

--- GA Final Allocations ---
EV # | Preferred (kW) | Allocated (kW)
   1 |           6.34 |           5.81
   2 |           8.76 |           8.24
   3 |           3.00 |           2.48
   4 |           5.42 |           4.90
   5 |           4.17 |           3.65
   6 |           3.74 |           3.28
   7 |           4.49 |           3.97
   8 |           5.76 |           5.24
   9 |           6.17 |           5.65
  10 |           7.31 |           6.79
--------------------------------------------------------------------
Sum of Preferred Rates: 55.17 kW
Sum of Allocated Rates (GA): 50.00 kW (Target: 50.00 kW)
Objective value (sum of squared differences) for GA solution: 2.6765
Analytical optimal objective value: 2.6732
Analytical sum: 50.0000
GA plots for best fitness and stopping criteria should have been displayed in separate figures.
```

![](ga_iter.png)

![](ga.png)

---

## Discussion

本稿では、EV充電リソース割り当て問題に対し、分散型勾配降下法（DGD）、交替方向乗数法（ADMM）、粒子群最適化（PSO）、および遺伝的アルゴリズム（GA）の4つの異なる手法を適用しました。

### 1. 分散型勾配降下法 (Distributed Gradient Descent - DGD)

- **全体的な振る舞い：** DGDは、局所的な勾配情報とコンセンサス平均化を用いて、各EVの推定値を希望充電率の平均値に近づけます。この平均値は、電力割り当てを決定するための解析的な式で使用されます。
- **長所：**
  - 分散方式で簡単に実装できます。
  - 平均化のためのコンセンサス部分はよく理解されています。
- **短所：**
  - 最終的な割り当てステップ `p_i = p_pref_i + P_total/N - z_i`（ここで `z_i` は推定平均値）は、`z_i` が完全に収束していない場合や、境界 `p_i >= 0` が有効になった場合に、`sum(p_i) = P_total` を本質的に保証しません。
  - 制約を満たすために事後的なスケーリング（`max(0, ...)` および比例スケーリング）が必要であり、これにより解が真の最適解から逸脱する可能性があります。
- **結果：** **可**。分散型アプローチを提供しますが、厳密な制約充足のためにはヒューリスティックに依存します。解の品質は、コンセンサスの精度とヒューリスティックスケーリングの影響に左右されます。

---

### 2. 交替方向乗数法 (Alternating Direction Method of Multipliers - ADMM)

- **全体的な振る舞い：** ADMMは問題を、局所的なEVサブ問題（価格シグナルを考慮した二次コストの最小化）と、大域的な制約 `sum(p_i) = P_total` を強制するために分散型コンセンサスを使用する価格更新メカニズムに分解します。
- **長所：**
  - **総電力制約**と非負制約を効果的に満たします。
  - この凸問題に対して、解析的なKKT最適解に非常に近い解に収束します。
  - 分散型制約付き最適化のための堅牢なフレームワークです。
- **短所：**
  - ネストされたコンセンサスループ（合計推定と価格合意のため）と慎重なパラメータ調整（`rho`, `beta`）のため、DGDよりも実装が複雑です。
  - ADMMイテレーション内のコンセンサスが十分なステップ実行されない場合や、ADMMパラメータが適切に調整されていない場合、収束が遅くなることがあります。
- **結果：** **優（分散型手法のチャンピオン）**。ADMMは、この問題に対して、全ての制約を確実に強制し、真に分散化された方法でほぼ最適な割り当てを達成します。

---

### 3. 粒子群最適化 (Particle-Swarm Optimization - PSO)

- **全体的な振る舞い：** 大域的最適化メタヒューリスティックであるPSOは、この問題に集中的に適用されました。等式制約 `sum(p_i) = P_total` は目的関数内のペナルティ項を使用して処理されました。境界制約（`p_i >= 0`, `p_i <= ub_i`）は直接処理されました。
- **長所：**
  - 微分不要で、複雑な目的関数にも容易に適用できます。
  - 探索空間の大域的な探索に適しています。
- **短所：**
  - **ペナルティによる等式制約の扱いの難しさ**：解が `sum(p_i) = P_total` を遵守するかどうかは、ペナルティ係数とPSOがペナルティ項を最小化する能力に大きく依存します。多くの場合、これによりわずかな違反が生じ、事後的な正規化が必要になります。
  - 解の品質は、ペナルティ係数とPSOパラメータに敏感である可能性があります。
  - 得られた解は、特にペナルティが探索状況を歪める場合、ADMMやGAほど真の最適解に近くないかもしれません。
- **結果：** **劣～可**。標準的なPSOは、特殊なメカニズム（修復オペレータなど）や非常に慎重なペナルティ調整なしでは、正確な等式制約の充足に苦労します。事後的な正規化の必要性が欠点です。

---

### 4. 遺伝的アルゴリズム (Genetic Algorithm - GA)

- **全体的な振る舞い：** もう一つの大域的最適化メタヒューリスティックであるGAも、集中的に適用されました。重要なことに、MATLABの `ga` は、境界制約に加えて、線形等式制約（`Aeq*x = beq`）を直接処理できます。
- **長所：**
  - **線形等式制約の効果的な処理**：制約 `sum(p_i) = P_total` は、通常 `ConstraintTolerance` の範囲内で高い精度で満たされました。
  - 微分不要で、大域的な探索に適しています。
  - 解析的な最適値に非常に近い目的関数値を達成しました。
- **短所：**
  - この特定の凸問題に対しては、特殊な凸ソルバーやADMMよりも計算量が多くなる可能性があります。
  - 性能はGAパラメータの調整に依存する可能性があります（ただし、デフォルトは多くの場合堅牢です）。
  - メタヒューリスティックとして、大域的最適解への確率的な収束を提供します。
- **結果：** **非常に良い**。GAは、組み込みのメカニズムのおかげで、特に困難な合計制約を含む全ての制約を尊重する高品質な解を提供しました。直接的な制約処理が利用可能な場合、強力な大域的最適化の代替手段となります。

---

### まとめ (Takeaway)

- **線形制約を持つ凸二次計画リソース割り当て問題の場合：**
  - **分散型手法**：ADMMは非常に効果的であり、分散化された方法で制約を厳密に強制しながら、ほぼ最適な解を提供します。
  - **集中型手法**：`quadprog` のようなソルバー（図示されていませんが、この特定の問題タイプを一元的に解く場合に理想的）が最も効率的です。一般的な大域的オプティマイザの中では、**遺伝的アルゴリズム（GA）**は線形等式制約を直接処理できるため、そのような制約に対してより単純なペナルティ法に依存するPSOよりも優れた性能を示します。
- DGDのようなより単純な分散型手法は、基本的な調整や、制約充足のための何らかのヒューリスティックな事後処理が許容できる場合に役立ちます。
- PSOのような標準的なメタヒューリスティックは、厳しい等式制約のある問題を確実に解くために、慎重な適応（例：特殊な制約処理技術、修復オペレータ、またはハイブリッド化）が必要です。基本的なペナルティ関数だけに頼るのでは、多くの場合、精度が不十分です。
- ソルバーの選択は、問題の性質（凸性、微分可能性）、分散型解と集中型解のどちらが必要か、制約の複雑さ、および要求される精度に依存します。この特定のEV充電問題では、ADMM（分散型）とGA（集中型大域的）の両方が優れた結果をもたらしましたが、ADMMの方が「分散最適化」のテーマにより適合していました。

## References

2. D. P. Bertsekas, A. Nedić, and A. Ozdaglar, *Convex Analysis and Optimization*, Athena Scientific.

3. **Average Consensus Problem** 、*Distributed Multi-Agent Optimization via Dual Decomposition* (Section 2.8) [Diva Portal](https://www.diva-portal.org/smash/get/diva2%3A453798/FULLTEXT01.pdf) 。

4. A. Nedić and A. Ozdaglar, “Distributed Subgradient Methods for Multi‐Agent Optimization,” *IEEE Trans. Autom. Control*, 54(1): 48–61, 2009 [SCIRP](https://www.scirp.org/%28S%28czeh2tfqw2orz553k1w0r45%29%29/reference/referencespapers?referenceid=3220708) 。

5. A. Nedić and A. Ozdaglar, “Distributed Subgradient Methods for Multi-Agent Optimization” (最終稿), MIT [Massachusetts Institute of Technology](https://web.mit.edu/Asuman/Desktop/asuman/www/documents/distributed-journal-final.pdf) 。

6. S. Boyd et al., “Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers,” *Foundations and Trends in Machine Learning*, 3(1): 1–122, 2011 [Stanford University](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf) 。

7. **Consensus ALADIN** 非凸コンセンサス最適化のためのフレームワーク, Xu Du et al., 2023 [arXiv](https://arxiv.org/pdf/2306.05662) 。

8. G. Buzzard et al., “Optimization-Free Reconstruction Using Consensus Equilibrium,” SIIMS, 2019 [Purdue Engineering](https://engineering.purdue.edu/~bouman/Plug-and-Play/webdocs/SIIMS01.pdf) 。

9. マルチエージェント制御におけるイベントトリガー更新を伴う分散時変最適化 [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006925) 。

10. 多目的問題のための多層分散コンセンサスアルゴリズム [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306261921007923) 。

11. **Subgradient Method** 、Wikipedia、2024年 [en.wikipedia.org](https://en.wikipedia.org/wiki/Subgradient_method) 。