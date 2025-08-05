# 分散最適化：概要とMATLABによるトラック隊列走行の例

## 目次

1. [分散最適化の概要](#分散最適化の概要)
   * [一般的概念](#一般的概念)
   * [分散最適化の基礎](#分散最適化の基礎)
   * [スウォームインテリジェンス・メタヒューリスティクス](#スウォームインテリジェンスメタヒューリスティクス)
   * [スウォーム／マルチエージェント制御](#スウォームマルチエージェント制御)
   * [主要なトレンドと未解決の課題](#主要なトレンドと未解決の課題)
   * [まとめ](#まとめ-1)
2. [応用例：トラック隊列走行のための分散型モデル予測制御 (DMPC)](#応用例トラック隊列走行のための分散型モデル予測制御-dmpc)
   * [問題設定](#問題設定)
   * [DMPC手法の概要](#dmpc手法の概要)
   * [MATLABスクリプトの解説](#matlabスクリプトの解説)
     * [シミュレーションと物理パラメータ](#シミュレーションと物理パラメータ)
     * [MPCパラメータと重み係数](#mpcパラメータと重み係数)
     * [初期状態と目標速度プロファイル](#初期状態と目標速度プロファイル)
     * [メインシミュレーションループ](#メインシミュレーションループ)
     * [各トラックの局所MPC問題](#各トラックの局所mpc問題)
     * [MPC目的関数 (`mpcObjectiveTruck_i`)](#mpc目的関数-mpcobjectivetruck_i)
     * [MPC制約関数 (`mpcConstraintsTruck_i`)](#mpc制約関数-mpcconstraintstruck_i)
   * [この概念的スクリプトの限界と今後の展望](#この概念的スクリプトの限界と今後の展望)
3. [参考文献](#参考文献)

## 分散最適化の概要

### 一般的概念

```sql
1. Global Optimization Problem (グローバル最適化問題)
   F(x₁, x₂, …, x_N)
   └─► N個のエージェントを協調させることによりFの最小化/最大化を目指す
      (Aim to min/max F by coordinating N agents)

2. Multi-Agent System (マルチエージェントシステム)
   ├─ Agent 1 (エージェント1)
   ├─ Agent 2 (エージェント2)
   ├─ …
   └─ Agent N (エージェントN)

3. Agent 1 Workflow (Representative Example) (エージェント1のワークフロー - 代表例)
   ├─ [Local Data/Knowledge] (ローカルデータ/知識)
   │     • e.g., p_pref₁, local constraints (例: p_pref₁, ローカル制約)
   ├─ [Local Objective/Cost] (ローカル目的/コスト)
   │     • e.g., (p₁ – p_pref₁)² or local part of F (例: (p₁ – p_pref₁)² または F のローカル部分)
   ├─ [Local Computation Engine] (ローカル計算エンジン)
   │     • Inputs: x₁(k), local objective (入力: x₁(k), ローカル目的)
   │     └─► Method Choice: (手法の選択)
   │          a) Gradient-Based (勾配ベース)
   │             1. Compute ∇f₁(x₁(k)) (∇f₁(x₁(k)) を計算)
   │             2. Update using ∇f₁ and messages (∇f₁ とメッセージを使用して更新)
   │          b) Gradient-Free (勾配フリー)
   │             1. Evaluate f₁(x₁(k)) (f₁(x₁(k)) を評価)
   │             2. Heuristic update + messages (ヒューリスティック更新 + メッセージ)
   ├─ [Prepare & Send Message] (メッセージの準備と送信)
   │     • Data depends on method (state, gradient, bid, etc.) (データは手法に依存 (状態、勾配、入札など))
   ├─ [Communication Network] (通信ネットワーク)
   │     • Topology: Ring / Star / Mesh / Complete (トポロジー: リング / スター / メッシュ / 完全グラフ)
   │     • Carries messages among neighbors (隣接エージェント間でメッセージを運ぶ)
   ├─ [Receive Messages] (メッセージの受信)
   │     • From neighbors (隣接エージェントから)
   ├─ [Update State] (状態の更新)
   │     • New estimate x₁(k+1) (新しい推定値 x₁(k+1))
   └─ [Iterate k→k+1] (反復 k→k+1)

4. Agent 2 Workflow (エージェント2のワークフロー)
   (Same steps as Agent 1, uses own data/objective) (Agent 1 と同じステップ、自身のデータ/目的を使用)

5. …

6. Agent N Workflow (エージェントNのワークフロー)
   (Same pattern) (同じパターン)

7. What’s Exchanged on the Network (ネットワーク上で交換されるもの)
   – Current states/estimates x_j(k) (現在の状態/推定値 x_j(k))
   – Local gradients ∇f_j (ローカル勾配 ∇f_j)
   – Dual variables / prices λ_j (双対変数 / 価格 λ_j)
   – Proposed actions or bids (提案されたアクションまたは入札)
   – Best-known solutions (for meta-heuristics) (既知の最良解 (メタヒューリスティクス用))
   – Aggregated values (sums, averages) (集計値 (合計、平均))
```

**図とマルチエージェント最適化プロセスの説明:**

1. **Global Optimization Problem (グローバル最適化問題) ($F(x_1, ..., x_N)$):**
   
   * これは、エージェントのシステム全体が達成しようとする包括的な目標です。
   * 多くの場合、すべてのエージェントの変数/アクションの関数です (例: 総コストの最小化 $\sum f_i(x_i)$、またはコンセンサス $x_1 = x_2 = ... = x_N$ の達成)。
   * エージェントの最適なアクションが他のエージェントのアクションや共有制約に依存する場合があるため、解決にはしばしば調整が必要です。

2. **Multi-Agent System (マルチエージェントシステム):**
   
   * 複数の自律的な `Agent` (エージェント1, エージェント2, ..., エージェントN) で構成されます。
   * 各エージェント `i` は、独自の決定変数 $x_i$ を持ちます。

3. **Individual Agent (e.g., Agent 1) (個々のエージェント - 例：エージェント1):**
   
   * **Local Data/Knowledge (ローカルデータ/知識):** 各エージェントはプライベートな情報を持っています (例: 優先充電レート $p_{pref_i}$、ローカルセンサーの読み取り値、ローカル制約)。
   * **Local Objective/Cost (ローカル目的/コスト):** ローカルデータとグローバルな目的に基づいて、各エージェントはしばしば最適化するためのローカル目的関数 $f_i(x_i)$ を持つか、グローバルな目的に一部貢献します。時には、ローカル目的には、結合や合意を強制するために隣人の状態に関連する項も含まれます。
   * **Agent State/Estimate ($x_i(k)$) (エージェントの状態/推定値):** 各反復 $k$ で、エージェントは決定変数の現在の値またはグローバルな量の推定値を維持します。
   * **Local Computation Engine (ローカル計算エンジン):** これはエージェントの意思決定の中核です。
     * 現在の状態 $x_i(k)$、ローカル目的 $f_i$、ローカルデータ、および他のエージェントから受信したメッセージを入力として受け取ります。
     * 次に、`Local Method` (ローカル手法) を適用して、次のアクションを計算するか、状態を更新します。
   * **Local Method (ローカル手法):**
     * **Gradient-Based (勾配ベース):**
       * ローカル目的 $f_i$ が微分可能な場合、エージェントはローカル勾配 $\nabla f_i(x_i(k))$ を計算できます。
       * 更新ルールは通常、この負の勾配に沿ったステップ (最小化の場合) と隣人からの情報 (例: DGDにおける隣人の状態の加重平均、またはADMMにおける双対変数に基づく更新) を組み合わせます。
       * 例: 分散勾配降下法 (DGD)、ADMMサブプロブレム、双対分解法。
     * **Gradient-Free (勾配フリー):**
       * 勾配が利用できない、計算コストが高すぎる、または問題が微分不可能またはブラックボックスである場合に使用されます。
       * エージェントは、現在の状態 (または提案された状態) でローカル目的 $f_i(x_i(k))$ を評価します。
       * 更新ルールは、特定のアルゴリズムから導出されたヒューリスティックな原則に従います (例: PSO粒子は、自身のパーソナルベストと近傍/グローバルベストに基づいて速度と位置を更新します。GA個体は選択、交叉、突然変異を受けます)。
       * 例: 粒子群最適化 (PSO)、遺伝的アルゴリズム (GA)、アリコロニー最適化 (ACO)、シミュレーテッドアニーリング (分散設定に適合させた場合)。
   * **Prepare/Send Message (メッセージの準備と送信):** ローカル計算の後、エージェントは隣人に送信するための関連情報を含むメッセージを準備します。
   * **Receive Messages (メッセージの受信):** 次の計算サイクルの前に、エージェントは隣人からメッセージを受信します。
   * **New State/Estimate ($x_i(k+1)$) (新しい状態/推定値):** ローカル計算と受信したメッセージに基づいて、エージェントは状態を更新します。
   * **Iterate ($k \rightarrow k+1$) (反復):** 設定された反復回数または収束基準が満たされるまで、プロセスが繰り返されます。1回の反復で受信されたメッセージは、通常、*次*の反復の計算に使用されます。

4. **Communication Network (通信ネットワーク):**
   
   * どのエージェントが互いにメッセージを交換できるかを定義します。
   * **Topology (トポロジー):** 完全グラフ (全対全)、リング、スター、メッシュ、またはその他のグラフ構造にすることができます。トポロジーは、収束速度と堅牢性に大きな影響を与えます。
   * 通信は通常、不完全であると想定されます (より高度なモデルでは、遅延、ノイズ、リンク障害)。ここでは簡単のため、信頼性の高い隣人間通信を想定しています。

5. **Information Exchanged (What is Communicated) (交換される情報):**
   
   * メッセージの内容は、特定の分散アルゴリズムに大きく依存します。
     * **Current states/estimates ($x_j(k)$) (現在の状態/推定値):** コンセンサスアルゴリズムで一般的です (DGDのように、エージェントが隣人の状態を平均化する場合など)。
     * **Local gradients ($\nabla f_j$) (ローカル勾配):** 一部のアルゴリズムでは勾配情報を交換する場合があります。
     * **Dual variables / Prices ($\lambda_j$) (双対変数 / 価格):** ADMMおよび双対分解法の中核であり、共有制約に違反することの「価格」を表します。
     * **Proposed actions / Bids (提案されたアクション / 入札):** 市場ベースのメカニズムや一部のゲーム理論的アプローチで使用されます。
     * **Best-known solutions (既知の最良解):** 一部の分散メタヒューリスティクスで使用されます (例: PSOの粒子が自身のパーソナルベスト位置を共有したり、近傍のベストに関する情報を受信したりする)。
     * **Aggregated values (集計値):** エージェントは、特定のローカル値の合計や平均を協調して計算する場合があります。

**Overall Flow (全体的な流れ):**

システムは通常、反復的なラウンドで動作します。

1. **Initialization (初期化):** エージェントは初期状態 $x_i(0)$ から開始します。
2. **Iteration $k$ (反復 $k$):**
   a.  **Local Computation (ローカル計算):** 各エージェント $i$ は、現在の状態 $x_i(k)$ および前のラウンド (または反復 $k-1$ から) で受信したメッセージを使用して、選択した手法 (勾配ベースまたは勾配フリー) に従ってローカル計算を実行します。
   b.  **State Update (状態更新):** エージェントは状態を $x_i(k+1)$ に更新します。
   c.  **Communication (通信):** 各エージェント $i$ は、メッセージ ($x_i(k+1)$ の一部またはその他の派生情報を含む) を通信ネットワークを介して隣人に送信します。
3. **Repeat (繰り返し):** $k$ をインクリメントし、収束または最大反復回数に達するまでステップ2aに進みます。

目標は、エージェントの状態 $x_i$ が集合的にグローバルな最適化問題 $F$ の解 (または良好な近似解) に収束することです。

### 分散最適化の基礎

- **Consensus‐based Gradient Methods (コンセンサスベース勾配法)**
  
  - **Distributed Gradient Descent (DGD) (分散勾配降下法)** およびその高速化版（例：EXTRA, DIGing, NIDS）は、ローカルな勾配ステップとコンセンサス（平均化）を組み合わせることで、ネットワーク上の（強）凸問題を解きます。
  - **Gradient Tracking (勾配追跡)** メソッドは、各エージェントでグローバルな勾配推定値を追跡し、より速い収束と時変グラフに対するロバスト性をもたらします。

- **Operator Splitting & ADMM (作用素分割とADMM)**
  
  - **Decentralized ADMM (分散ADMM)** は、交互方向乗数法 (Alternating Direction Method of Multipliers) をピアツーピア方式で適用し、グローバルな目的関数をローカルなサブ問題とコンセンサス制約に分割します。
  - 最近のバリアントは、メッセージングのオーバーヘッドを削減するために、**非同期 (asynchronous)** 、**イベントトリガー型 (event‐triggered)** 、または **通信圧縮型 (communication‐compressed)** の更新に焦点を当てています。

- **Primal–Dual & Second-Order Methods (主双対法と二次法)**
  
  - **Primal–Dual Algorithms (主双対アルゴリズム)** （例：Chambolle–Pock スタイル）はネットワークに適応されており、鞍点問題に対する高速な収束率を提供します。
  - **Distributed Newton / Quasi-Newton (分散ニュートン法／準ニュートン法)** メソッドは、局所的に曲率情報を近似することで、悪条件の問題に対する収束を高速化し、多くの場合、追加の計算と引き換えに通信ラウンド数を削減します。

---

### スウォームインテリジェンス・メタヒューリスティクス

- **Particle Swarm Optimization (PSO) and Variants (粒子群最適化とその亜種)**
  
  - **Quantum-behaved PSO (QPSO) (量子挙動PSO)** 、**Bare-Bones PSO** 、および **Fully Informed PSO** は、探索と活用のトレードオフを改善します。
  - **Multi‐swarms (マルチスウォーム)** は、大規模な探索を相互作用するサブスウォームに分解し、各サブスウォームが問題の一部を最適化したり、異なる領域を探索したりします。

- **Ant Colony Optimization (ACO) (アリコロニー最適化)**
  
  - **Max–Min ACO** は、早期収束を避けるためにフェロモンに明示的な境界を設定します。
  - **Hybrid ACO-PSO (ACOとPSOのハイブリッド)** や **ACO-GA (ACOとGAのハイブリッド)** は、相補的なメカニズム（交叉/突然変異または速度更新）を利用して局所最適解から脱出します。

- **Other Bio-inspired Techniques (その他の生物模倣技術)**
  
  - **Artificial Bee Colony (ABC) (人工蜂コロニー)** 、**Bacterial Foraging Optimization (BFO) (バクテリア採餌最適化)** 、および **Firefly Algorithm (ホタルアルゴリズム)** は、動的なパラメータ適応、並列集団、およびレヴィフライトステップによって調整され、高次元で多峰性の目的関数に対するパフォーマンスを向上させます。

---

### スウォーム／マルチエージェント制御

- **Consensus & Formation Control (合意とフォーメーション制御)**
  
  - **Leaderless Consensus (リーダーレス合意)** プロトコルは、ローカルな隣接エージェントとの通信のみにもかかわらず、すべてのエージェントが状態（例：フォーメーションの重心、ランデブーポイント）について合意することを保証します。
  - **Leader–Follower (リーダー・フォロワー)** アーキテクチャは、全体的な動きを操縦する情報を持つ「リーダー」エージェントのサブセットと、形状を維持するためにコンセンサスを実行するフォロワーを組み合わせます。

- **Distributed Model Predictive Control (DMPC) (分散モデル予測制御)**
  
  - エージェントは、ADMM や双対分解を介して結合制約（例：衝突回避、リソース共有）を持つローカルな MPC 問題を解き、限られた通信で集中型に近いパフォーマンスを達成します。

- **Event-Triggered & Self-Triggered Control (イベントトリガー制御と自己トリガー制御)**
  
  - ネットワーク負荷を軽減するために、エージェントは特定状態またはエラーベースの閾値を超えた場合にのみ情報を交換し、大幅に少ないメッセージで安定性とパフォーマンスを保証します。

- **Learning-Enhanced Control (学習強化型制御)**
  
  - **Graph Neural Networks (GNNs) (グラフニューラルネットワーク)** および **Multi-Agent Reinforcement Learning (MARL) (マルチエージェント強化学習)** は、データから直接制御ポリシーを調整することを学習し、非線形ダイナミクスや部分的観測可能性に対応します。

---

### 主要なトレンドと未解決の課題

- **Communication Efficiency (通信効率)** : 帯域幅を削減するための圧縮メッセージ、量子化更新、および散発的/イベントトリガー型スキーム。
- **Robustness & Privacy (堅牢性とプライバシー)** : リンク障害、敵対的エージェントを許容し、差分プライバシーを保証するアルゴリズム。
- **Scalability (スケーラビリティ)** : 階層的スウォーム、部分空間分解、または平均場近似を介した数千のエージェントの処理。
- **Nonconvex & Time-Varying Objectives (非凸および時変目的)** : 非凸ランドスケープ（ロボティクスや深層学習で一般的）および動的に変化するネットワークトポロジーにおける収束の保証。
- **Bridging Meta-heuristics & Convex Methods (メタヒューリスティクスと凸最適化手法の橋渡し)** : より良いグローバル探索のために、高速なコンセンサスベースの凸ソルバーをグローバルなメタヒューリスティックの外側ループ内に埋め込むハイブリッドフレームワーク。

---

### まとめ {#まとめ-1}

マルチエージェントおよびスウォームコンテキストにおける最先端の分散最適化は、（証明可能な保証を持つ）**厳密な凸/ADMMスタイルアルゴリズム**と、（困難で非凸なランドスケープのための）**生物に着想を得たメタヒューリスティクス**を相乗させ、現代のスウォーム制御は**学習駆動型**、**通信疎**、かつ**ロバスト**な協調法則を推進しています。この融合により、大規模なロボットチーム、センサーネットワーク、および分散機械学習システムが、現実的なネットワークおよびエージェントの制約下で効率的に動作することが可能になります。

---

## 応用例：トラック隊列走行のための分散型モデル予測制御 (DMPC)

このセクションでは、トラックの隊列走行シナリオに分散型モデル予測制御 (DMPC) を適用する方法について詳しく説明します。各トラックは独立したエージェントとして動作し、自身のローカルな最適化問題を解きながら、前方車両との協調を通じて隊列全体の目標（燃料消費削減、安全性、基準速度追従）を達成しようとします。

### 問題設定

考慮するシステムは、$N$台のトラックからなる隊列です。主な目標は以下の通りです。

1. **リーダー追従:** 隊列の先頭車両（リーダー）は、与えられた目標速度プロファイル $v_{ref}(t)$ に可能な限り正確に追従する。

2. **車間距離維持:** 各追従車両（フォロワー）は、前方の車両との間に安全かつ効率的な車間距離を維持する。この車間距離は、一定の最小ギャップ $s_{min\_gap\_const}$ と、フォロワー自身の速度 $v_i(t)$ に比例する時間ヘッドウェイ $t_h$ に基づいて決定されることが多い。
   
   $$
   \text{目標バンパー間ギャップ}_i(t) = s_{min\_gap\_const} + t_h \cdot v_i(t)
   $$

3. **燃料消費最小化:** 隊列全体の総燃料消費量、あるいは各トラックの燃料消費量を最小化する。空気抵抗は速度の2乗に比例し、後続車両は先行車両によるスリップストリーム効果（遮蔽効果）の恩恵を受けるため、これが重要な要素となる。

4. **安全性と快適性:** 各トラックは、自身の物理的な速度制限 $v_{min}, v_{max}$ および加速度制限 $a_{min}, a_max$ を遵守する。また、急激な加速度変化（ジャーク）を避けることで、乗り心地や積荷の安全性を向上させる。

これらの目標はしばしば相反するため、重み付けによってバランスを取る必要があります。

### DMPC手法の概要

分散型モデル予測制御 (DMPC) では、集中型MPCとは異なり、システム全体を単一の最適化問題として解くのではなく、各エージェント（この場合は各トラック）が自身のローカルな最適化問題を繰り返し解きます。

1. **エージェントごとのMPC:** 各トラック $i$ は、自身のMPCコントローラを持ちます。
2. **予測ホライズン:** 各トラック $i$ は、現在の時刻 $k$ から $H_p$ ステップ先までの将来の挙動を予測します。
3. **ローカル最適化:** 各トラック $i$ は、以下の要素を考慮して、自身の将来の加速度シーケンス $\{a_i(k), a_i(k+1), \dots, a_i(k+H_p-1)\}$ を決定します。
   * **ローカルな目的関数 $J_i$:**
     * リーダーの場合：$v_{ref}$ との速度誤差、燃料消費、加速度の滑らかさ。
     * フォロワーの場合：前方車両との速度誤差、前方車両との目標車間距離からの誤差、燃料消費、加速度の滑らかさ。
   * **ローカルな制約条件:**
     * 車両ダイナミクス: $s_i(j+1) = s_i(j) + v_i(j) \Delta t + \frac{1}{2} a_i(j) \Delta t^2$, $v_i(j+1) = v_i(j) + a_i(j) \Delta t$.
     * $v_{min} \le v_i(j) \le v_{max}$.
     * $a_{min} \le a_i(j) \le a_{max}$.
     * 車間距離制約 (フォロワーのみ): $s_{i-1}(j) - s_i(j) \ge L_{truck} + s_{min\_gap\_const} + t_h v_i(j)$ (車頭間)。
4. **情報交換:** フォロワー $i$ が自身の最適化問題を解くためには、先行車両 $i-1$ の予測される将来の状態（位置 $s_{i-1}$、速度 $v_{i-1}$）に関する情報が必要です。この情報は、通信を介して取得されるか、仮定に基づいて推定されます。
5. **逐次解法 (本スクリプト):** リーダーが最初に自身のMPC問題を解き、その結果（またはその一部）が最初のフォロワーに渡され、最初のフォロワーが問題を解き…というように、隊列後方に向かって情報が伝播し、各トラックが順番に最適化を行います。
6. **後退ホライズン:** 各トラックは、計算された最適な加速度シーケンスのうち、最初のステップ $a_i(k)$ のみを実施します。次の制御時刻 $k+1$ で、全体のプロセスが繰り返されます。

### MATLABスクリプトの解説

提供されたMATLABスクリプトは、このDMPCの概念を実装したものです。以下に主要な部分を解説します。

#### シミュレーションと物理パラメータ

スクリプトの冒頭では、シミュレーション時間 `T_simulation_sec`、制御時間ステップ `dt`、トラックの台数 `N`、トラックの物理的特性（全長 `L_truck`、質量 `m`、空気抵抗係数 `C_d_truck`、前面投影面積 `A_f`、転がり抵抗係数 `C_r`）、環境パラメータ（空気密度 `rho_air`、重力加速度 `g`）、駆動系パラメータ（効率 `eta_drivetrain`、アイドル時出力 `P_idle`）、燃料消費係数 `k_fuel_g_per_J` が定義されます。
また、後続車両の空気抵抗を低減する遮蔽効果 `shielding_factors` も考慮されます。

```matlab
% === シミュレーションパラメータ ===
T_simulation_sec = 60; % 総シミュレーション時間（秒）
dt = 1.0;              % コントローラの更新間隔（秒）
num_sim_steps = T_simulation_sec / dt;  % シミュレーションステップ数

% === 車両・プラトーンの物理パラメータ ===
N = 3;                   % トラックの台数
L_truck = 18;            % 各トラックの全長 [m]
% ... (その他の物理パラメータ) ...
shielding_factors = zeros(N,1);
if N > 1, shielding_factors(2) = 0.3; end
if N > 2, shielding_factors(3) = 0.4; end
C_d_eff = C_d_truck * (1 - shielding_factors); % 有効空力抵抗係数
```

車両の運用上の制約として、最小・最大速度 (`v_min`, `v_max`)、最小・最大加速度 (`a_min`, `a_max`)、および車間距離ポリシーに関連する定数 (`s_min_gap_const`, `t_h`) が設定されます。

#### MPCパラメータと重み係数

各トラックのMPCコントローラは、予測ホライズン `Hp` を持ちます。これは、何ステップ先まで予測して最適化を行うかを示します。
目的関数内の各項目の相対的な重要度を調整するために、重み係数が定義されます。

```matlab
% === MPC予測ホライズン ===
Hp = 10;  % 予測ホライズンのステップ数

% === 重み係数の設定 ===
w_v_ref_leader = 1.0;       % リーダーの目標速度追従重み
w_v_track_follower = 0.8;   % フォロワーの先行車速度追従重み
w_gap_control = 1.0;        % 車間距離制御重み
w_fuel_leader = 0.01;       % リーダーの燃料消費重み
w_fuel_follower = 0.05;     % フォロワーの燃料消費重み
w_accel_smooth = 0.01;      % 加速度平滑化重み
```

これらの重みの調整は、DMPCの性能に大きく影響します。

#### 初期状態と目標速度プロファイル

シミュレーション開始時の各トラックの位置 `s_platoon(:,1)` と速度 `v_platoon(:,1)` が設定されます。初期位置は、定義された車間距離ポリシーに従って、衝突しないように配置されます。
リーダーが追従すべき目標速度プロファイル `v_ref_profile` も定義されます。このスクリプトでは、加速・定速・減速の区間を持つ台形型のプロファイルが例として使用されています。

```matlab
% === リーダー用の目標速度プロファイル ===
v_ref_profile = ones(num_sim_steps + Hp, 1) * 20;
if num_sim_steps + Hp >= 20
    v_ref_profile(1:10) = linspace(15, 22, 10);
    v_ref_profile(11:num_sim_steps+Hp-10) = 22;
    v_ref_profile(num_sim_steps+Hp-9:num_sim_steps+Hp) = linspace(22, 18, 10);
end
v_ref_profile = min(max(v_ref_profile, v_min), v_max);

% === 初期状態の定義（位置・速度）===
v_initial = v_ref_profile(1) * ones(N,1);
s_initial = zeros(N,1); s_initial(1) = 0;
for i = 2:N
    desired_gap_initial = L_truck + s_min_gap_const + t_h * v_initial(i);
    s_initial(i) = s_initial(i-1) - desired_gap_initial;
end
s_platoon(:,1) = s_initial;
v_platoon(:,1) = v_initial;
```

#### メインシミュレーションループ

シミュレーションは `k_sim = 1` から `num_sim_steps` まで実行されます。各 `k_sim` ステップで以下が行われます。

1. **現在状態の取得:** 各トラックの現在の位置 `current_s` と速度 `current_v` を取得します。
2. **トラックごとのMPC:** `i_truck = 1` (リーダー) から `N` (最後のフォロワー) まで、各トラックが順番に自身のローカルMPC問題を解きます。
   * リーダーの場合、グローバル目標速度 `v_ref_horizon` を参照します。
   * フォロワーの場合、先行車両 `i_truck-1` の予測軌道 (`predicted_s_ahead`, `predicted_v_ahead`) を取得します。このスクリプトでは、この予測は簡略化されています（例：先行車両が前ステップの加速度を維持すると仮定）。
   * `fmincon` を使用して、各トラック $i$ の目的関数 `objFunLocal` を、制約 `nonlconFunLocal` の下で最小化する最適な加速度シーケンス `a_i_optimal_sequence` を求めます。
   * 計算されたシーケンスの最初の加速度 `a_i_optimal_sequence(1)` のみが、そのトラックの今ステップの制御入力 `optimal_a_this_step(i_truck)` として採用されます。
3. **状態更新:** 全トラックの制御入力 `optimal_a_this_step` が決定された後、それらを適用して各トラックの次時刻の位置と速度 (`s_platoon(:, k_sim+1)`, `v_platoon(:, k_sim+1)`) を計算します。

```matlab
for k_sim = 1:num_sim_steps
    current_s = s_platoon(:, k_sim);
    current_v = v_platoon(:, k_sim);
    optimal_a_this_step = zeros(N,1);

    for i_truck = 1:N
        % ... (v_ref_horizon, a_prev_applied_this_truck, predicted_s_ahead の準備) ...

        s_i_current_val = current_s(i_truck);
        v_i_current_val = current_v(i_truck);
        % ... (m_i_val, C_d_eff_i_val の準備) ...

        objFunLocal = @(a_i_seq) mpcObjectiveTruck_i(a_i_seq, i_truck, N, Hp, dt, ...);
        nonlconFunLocal = @(a_i_seq) mpcConstraintsTruck_i(a_i_seq, i_truck, N, Hp, dt, ...);

        options_local_mpc = optimoptions('fmincon', ...);
        [a_i_optimal_sequence, ~] = fmincon(objFunLocal, a_guess_local, ...);

        optimal_a_this_step(i_truck) = a_i_optimal_sequence(1);
    end

    a_platoon_actual(:, k_sim) = optimal_a_this_step;
    v_platoon(:, k_sim+1) = v_platoon(:, k_sim) + optimal_a_this_step * dt;
    v_platoon(:, k_sim+1) = max(v_min, min(v_max, v_platoon(:, k_sim+1)));
    s_platoon(:, k_sim+1) = s_platoon(:, k_sim) + v_platoon(:, k_sim) * dt ...
        + 0.5 * optimal_a_this_step * dt^2;
end
```

#### 各トラックの局所MPC問題

各トラック $i$ が解く最適化問題は以下のように定式化されます。
**決定変数:**

$$
\mathbf{a}_i^* = \{a_i^*(k), a_i^*(k+1), \dots, a_i^*(k+H_p-1)\}
$$

**目的関数 $J_i(\mathbf{a}_i)$:**

$$
\min_{\mathbf{a}_i} J_i = \sum_{j=0}^{H_p-1} \left( w_{v,i} \cdot (\text{速度誤差}_i(k+j+1))^2 + w_{g,i} \cdot (\text{ギャップ誤差}_i(k+j+1))^2 + w_{f,i} \cdot \text{燃料}_i(k+j) + w_{a,i} \cdot (\text{加速度コスト}_i(k+j))^2 \right)
$$

ここで、

* 速度誤差は、リーダーの場合は $v_i - v_{ref}$、フォロワーの場合は $v_i - v_{i-1,pred}$ です。
* ギャップ誤差は、フォロワーの場合のみ定義され、$ (\text{実際のバンパー間ギャップ}_i - \text{目標バンパー間ギャップ}_i) $ です。
* 燃料コストは、物理モデルに基づいて計算されます。
* 加速度コストは、加速度の大きさや変化率を抑制します。
* $w_{v,i}, w_{g,i}, w_{f,i}, w_{a,i}$ は対応する重み係数です。

**制約条件:** 予測ホライズン $j=0, \dots, H_p-1$ にわたって、

1. 車両ダイナミクス:
   
   $$
   v_i(k+j+1) = v_i(k+j) + a_i(k+j) \Delta t
   $$
   
   $$
   s_i(k+j+1) = s_i(k+j) + v_i(k+j) \Delta t + \frac{1}{2} a_i(k+j) \Delta t^2
   $$

2. 速度制約: $v_{min} \le v_i(k+j+1) \le v_{max}$

3. 加速度制約: $a_{min} \le a_i(k+j) \le a_{max}$ (これは `fmincon` の `lb`, `ub` で扱われる)

4. 車間距離制約 (フォロワー $i>1$ のみ):
   
   $$
   s_{i-1,pred}(k+j+1) - s_i(k+j+1) \ge L_{truck} + s_{min\_gap\_const} + t_h v_i(k+j+1)
   $$

#### MPC目的関数 (`mpcObjectiveTruck_i`)

この関数は、トラック $i$ の与えられた加速度シーケンス `a_i_seq` に対する総コスト $J_i$ を計算します。
内部で、`a_i_seq` を用いて $H_p$ ステップ先までのトラック $i$ の位置 $s_i\_pred$ と速度 $v_i\_pred$ を予測します。
次に、予測された状態に基づいて、速度追従コスト、車間距離制御コスト（フォロワーのみ）、燃料消費コスト、および加速度平滑化コストを計算し、重み付けして合計します。

```matlab
function J_i = mpcObjectiveTruck_i(a_i_seq, i_truck, ..., a_prev_applied_i, ...)
    % ... 状態予測 s_i_pred, v_i_pred ...
    % ... 燃料消費 fuel_i_total の計算 ...

    cost_accel_smooth = ...; % 加速度平滑化コスト

    if i_truck == 1 % Leader
        cost_v_tracking = w_v_ref_L * sum((v_i_pred(2:end) - v_ref_leader_horizon(1:Hp)).^2);
        cost_fuel = w_fuel_L * fuel_i_total;
        cost_gap_control = 0;
    else % Follower
        cost_v_tracking = w_v_track_F * sum((v_i_pred(2:end) - v_ahead_pred(2:end)).^2);
        actual_bumper_gap = (s_ahead_pred(2:end) - L_truck) - s_i_pred(2:end);
        target_bumper_gap = s_min_gap_const_i + t_h_i * v_i_pred(2:end);
        cost_gap_control = w_gap_F * sum((actual_bumper_gap - target_bumper_gap).^2);
        cost_fuel = w_fuel_F * fuel_i_total;
    end
    J_i = cost_v_tracking + cost_gap_control + cost_fuel + cost_accel_smooth;
end
```

燃料消費は、空気抵抗 $F_{aero} = \frac{1}{2} \rho_{air} C_{d,eff,i} A_f v^2$、転がり抵抗 $F_{roll} = m_i g C_r$、慣性力 $F_{inertia} = m_i a_i$ から総走行抵抗 $F_{total}$ を求め、車輪出力 $P_{wheel} = F_{total} v$ を計算します。エンジン出力 $P_{engine}$ は、$P_{wheel}$ が正の場合は $P_{wheel}/\eta_{drivetrain} + P_{idle}$、負の場合は $P_{idle}$ となり、これに燃料消費係数 $k_{fuel\_g\_per\_J}$ を乗じて燃料消費率を求めます。

#### MPC制約関数 (`mpcConstraintsTruck_i`)

この関数は、トラック $i$ の与えられた加速度シーケンス `a_i_seq` に対して、不等式制約 $c \le 0$ と等式制約 $ceq = 0$ を評価します。
`mpcObjectiveTruck_i` と同様に、まず状態を予測します。その後、予測された速度が $v_{min}, v_{max}$ の範囲内にあるか、また、フォロワーの場合は先行車両との車頭間距離が最小安全距離（ポリシーに基づく）以上であるかをチェックします。

```matlab
function [c, ceq] = mpcConstraintsTruck_i(a_i_seq, i_truck, ...)
    % ... 状態予測 s_i_pred, v_i_pred ...
    c = [];
    % 速度制約
    c = [c; v_i_pred(2:end) - v_max_i]; % v_i - v_max <= 0
    c = [c; v_min_i - v_i_pred(2:end)]; % v_min - v_i <= 0

    if i_truck > 1 % フォロワーの車間距離制約
        desired_front_to_front_sep = L_truck_i + s_min_gap_const_i + t_h_i * v_i_pred(2:end);
        actual_front_to_front_sep = s_ahead_pred(2:end) - s_i_pred(2:end);
        c_spacing = desired_front_to_front_sep - actual_front_to_front_sep; % 正なら制約違反
        c = [c; c_spacing];
    end
    ceq = []; % 等式制約なし
end
```

### この概念的スクリプトの限界と今後の展望

このスクリプトはDMPCの基本的な枠組みを示していますが、いくつかの重要な簡略化と限界があります。

1. **先行車両の予測:** フォロワーが使用する先行車両の将来軌道の予測は非常に単純です（例：一定加速度を仮定）。より高度なDMPCでは、先行車両が自身のMPCで計画した軌道（またはその一部）を通信で受信したり、より洗練された予測モデル（例：カルマンフィルタ）を使用したりします。
2. **通信:** 実際の通信遅延、パケットロス、帯域幅制限は考慮されていません。
3. **逐次解法と収束性:** 各時間ステップ内でトラックが順番に解を計算する逐次的なアプローチは、情報の流れが一方向的です。より協調的な解を得るためには、同じ時間ステップ内でトラック間で情報を交換し、ローカルな最適化を数回反復するようなアルゴリズム（例：ADMMやゲーム理論に基づくアプローチをDMPCに組み込む）が考えられます。これにより、システム全体としての収束性や性能が向上する可能性がありますが、計算負荷は増大します。
4. **安定性:** 分散制御システムの安定性を保証することは一般的に困難な課題です。このスクリプトでは、安定性に関する明示的な保証はありません。
5. **ソルバー:** 各ローカルMPC問題は `fmincon` (NLPソルバー) を使用しています。問題が二次計画問題(QP)として定式化できる場合（例えば、燃料モデルを二次関数で近似し、ダイナミクスを線形化するなど）、`quadprog` や専用のMPCソルバー（例：`mpcActiveSetSolver`、ただしこれはMATLAB MPC Toolboxの一部）を使用することで計算効率が向上する可能性があります。

今後の展望としては、これらの限界に対処するために、より現実的な通信モデルの導入、協調的な反復解法の検討、ロバスト性や安定性を考慮した制御設計、さらには機械学習を用いた予測モデルの改善などが考えられます。

---

## 参考文献

1. D. P. Bertsekas, A. Nedić, and A. Ozdaglar, *Convex Analysis and Optimization*, Athena Scientific.
2. **Average Consensus Problem** 、*Distributed Multi-Agent Optimization via Dual Decomposition* (Section 2.8) [Diva Portal](https://www.diva-portal.org/smash/get/diva2%3A453798/FULLTEXT01.pdf) 。
3. A. Nedić and A. Ozdaglar, “Distributed Subgradient Methods for Multi‐Agent Optimization,” *IEEE Trans. Autom. Control*, 54(1): 48–61, 2009 [SCIRP](https://www.scirp.org/%28S%28czeh2tfqw2orz553k1w0r45%29%29/reference/referencespapers?referenceid=3220708) 。
4. A. Nedić and A. Ozdaglar, “Distributed Subgradient Methods for Multi-Agent Optimization” (最終稿), MIT [Massachusetts Institute of Technology](https://web.mit.edu/Asuman/Desktop/asuman/www/documents/distributed-journal-final.pdf) 。
5. S. Boyd et al., “Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers,” *Foundations and Trends in Machine Learning*, 3(1): 1–122, 2011 [Stanford University](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf) 。
6. **Consensus ALADIN** 非凸コンセンサス最適化のためのフレームワーク, Xu Du et al., 2023 [arXiv](https://arxiv.org/pdf/2306.05662) 。
7. G. Buzzard et al., “Optimization-Free Reconstruction Using Consensus Equilibrium,” SIIMS, 2019 [Purdue Engineering](https://engineering.purdue.edu/~bouman/Plug-and-Play/webdocs/SIIMS01.pdf) 。
8. マルチエージェント制御におけるイベントトリガー更新を伴う分散時変最適化 [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006925) 。
9. 多目的問題のための多層分散コンセンサスアルゴリズム [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306261921007923) 。
10. **Subgradient Method** 、Wikipedia、2024年 [en.wikipedia.org](https://en.wikipedia.org/wiki/Subgradient_method) 。
