%% Decentralized Distributed MPC for Truck Platooning (トラック隊列走行のための分散型モデル予測制御)
% --------------------------------------------------------------------------
% DESCRIPTION: (概要)
% This script simulates a platoon of trucks where each truck attempts to
% optimize its trajectory in a decentralized and distributed manner using
% Model Predictive Control (MPC). The overall goal is to have the platoon
% move cohesively, with the leader tracking a reference velocity, followers
% maintaining safe and efficient spacing, and all trucks minimizing fuel.
% このスクリプトは、各トラックがモデル予測制御（MPC）を用いて分散型かつ協調的に
% 自身の軌道を最適化しようとするトラック隊列をシミュレートします。
% 全体的な目標は、リーダーが目標速度に追従し、フォロワーが安全かつ効率的な
% 車間距離を維持し、全トラックが燃料消費を最小限に抑えながら、隊列が
% 一体となって走行することです。
%
% METHODOLOGY: DECENTRALIZED DISTRIBUTED MODEL PREDICTIVE CONTROL (DMPC)
% (手法：分散型モデル予測制御 - DMPC)
%
% 1. Agent-Based Control: (エージェントベース制御)
%    - Each truck `i` in the platoon acts as an independent agent equipped
%      with its own MPC controller.
%    - 隊列内の各トラック`i`は、それぞれ独自のMPCコントローラを備えた独立した
%      エージェントとして動作します。
%
% 2. Local Optimization at Each Time Step: (各時間ステップでの局所最適化)
%    - At every control interval `dt_control`, each truck `i` solves a
%      finite-horizon optimal control problem for its own future actions.
%    - 各制御間隔`dt_control`において、各トラック`i`は自身の将来の行動に関する
%      有限ホライズン最適制御問題を解きます。
%    - Decision Variables (for truck `i`): A sequence of its own future
%      accelerations `a_i(k), a_i(k+1), ..., a_i(k+Hp-1)` over a
%      prediction horizon `Hp`.
%    - 決定変数（トラック`i`用）：予測ホライズン`Hp`にわたる自身の将来の
%      加速度シーケンス `a_i(k), a_i(k+1), ..., a_i(k+Hp-1)`。
%    - Local Objective Function (for truck `i`): This is a weighted sum of
%      truck-specific goals:
%    - 局所目的関数（トラック`i`用）：トラック固有の目標の重み付き合計です。
%        - Leader (Truck 1): (リーダー - トラック1)
%            - Track a global reference velocity profile (`v_ref_global`).
%            - グローバルな目標速度プロファイル（`v_ref_global`）に追従する。
%            - Minimize its own estimated fuel consumption.
%            - 自身の推定燃料消費量を最小化する。
%            - (Optional) Smooth its acceleration profile.
%            - (任意) 加速度プロファイルを滑らかにする。
%        - Followers (Truck `i > 1`): (フォロワー - トラック`i > 1`)
%            - Track the predicted velocity of the preceding truck (`i-1`).
%            -先行トラック（`i-1`）の予測速度に追従する。
%            - Maintain a target inter-truck gap (bumper-to-bumper) with
%              the preceding truck (`i-1`), based on a constant minimum
%              and a time-headway policy.
%            - 先行トラック（`i-1`）との目標車間距離（バンパー間）を維持する。
%              （一定の最小値とタイムヘッドウェイポリシーに基づく）
%            - Minimize its own estimated fuel consumption.
%            - 自身の推定燃料消費量を最小化する。
%            - (Optional) Smooth its acceleration profile.
%            - (任意) 加速度プロファイルを滑らかにする。
%    - Local Constraints (for truck `i`): (局所制約条件 - トラック`i`用)
%        - Its own vehicle dynamics (predicting future position and velocity
%          based on its planned accelerations).
%        - 自身の車両ダイナミクス（計画された加速度に基づく将来の位置と速度の予測）。
%        - Limits on its own acceleration (`a_min`, `a_max`).
%        - 自身の加速度の制限（`a_min`, `a_max`）。
%        - Limits on its own velocity (`v_min`, `v_max`).
%        - 自身の速度の制限（`v_min`, `v_max`）。
%        - For followers: A hard safety constraint on the minimum spacing
%          (front-to-front separation) with the preceding truck (`i-1`).
%        - フォロワー用：先行トラック（`i-1`）との最小車間距離（車両前端間）に関する
%          厳密な安全制約。
%    - Solver: Each truck's local optimization problem, being a Nonlinear
%      Program (NLP) due to the fuel model and dynamics, is solved using
%      MATLAB's `fmincon`.
%    - ソルバー：各トラックの局所最適化問題は、燃料モデルとダイナミクスによる
%      非線形計画問題（NLP）であるため、MATLABの`fmincon`を使用して解かれます。
%
% 3. Information Exchange and "Distributed" Nature: (情報交換と「分散型」の性質)
%    - The "distributed" aspect arises from how trucks share information to
%      make their local decisions.
%    - 「分散型」の側面は、トラックが局所的な決定を下すために情報を共有する方法から生じます。
%    - Preceding Vehicle Information: To achieve coordinated behavior, each
%      follower truck `i` requires the predicted future trajectory (position
%      and velocity over the horizon `Hp`) of the truck immediately
%      preceding it (`i-1`).
%    - 先行車両情報：協調的な挙動を達成するため、各フォロワートラック`i`は、
%      直前のトラック（`i-1`）の予測される将来の軌道（ホライズン`Hp`にわたる位置と速度）を必要とします。
%    - Communication Assumption (Simplified in this script): (通信の仮定 - このスクリプトでは簡略化)
%        - This script simulates a sequential solution: the leader solves its
%          MPC first, then the first follower, then the second, and so on,
%          all within the same simulation time step `k_sim`.
%        - このスクリプトは逐次解法をシミュレートします。つまり、同じシミュレーション
%          時間ステップ`k_sim`内で、リーダーが最初にMPCを解き、次に最初のフォロワー、
%          その次に2番目のフォロワー、という順序で解きます。
%        - When truck `i-1` solves its MPC, its planned trajectory could
%          ideally be communicated to truck `i`.
%        - トラック`i-1`がMPCを解くと、その計画された軌道は理想的にはトラック`i`に
%          通信される可能性があります。
%        - **Simplification in this Script:** For simplicity in this conceptual
%          outline, the prediction of the preceding truck's behavior is
%          rudimentary (e.g., assumes it continues its last applied
%          acceleration). A full DMPC would involve more sophisticated
%          management and use of these inter-agent predictions.
%        - **このスクリプトでの簡略化：** この概念的な概要では簡単にするため、
%          先行トラックの挙動の予測は初歩的なものです（例：最後に適用された加速度を
%          継続すると仮定）。完全なDMPCでは、これらのエージェント間予測のより
%          洗練された管理と使用が含まれます。
%    - No Central Coordinator for Trajectory Commands: Unlike the centralized
%      approach where one solver plans for everyone, here, each truck plans
%      for itself based on its local goals and information from its neighbors.
%    - 軌道コマンドのための中央コーディネーターは存在しません。1つのソルバーが
%      全員のために計画する集中型アプローチとは異なり、ここでは各トラックが
%      局所的な目標と隣接車両からの情報に基づいて自身のために計画します。
%
% 4. MPC Receding Horizon Principle: (MPC後退ホライズン原理)
%    - After solving its local optimization, each truck `i` applies only the
%      *first* acceleration command `a_i(k)` from its optimal sequence.
%    - 局所最適化を解いた後、各トラック`i`はその最適シーケンスから*最初*の
%      加速度コマンド`a_i(k)`のみを適用します。
%    - At the next time step `k_sim + 1`, the entire process (state update,
%      information gathering, local optimization) is repeated with updated
%      states and a shifted prediction horizon.
%    - 次の時間ステップ`k_sim + 1`では、プロセス全体（状態更新、情報収集、
%      局所最適化）が更新された状態とシフトされた予測ホライズンで繰り返されます。
%
% 5. Overall Behavior: (全体的な挙動)
%    - The collective behavior of the platoon emerges from these local,
%      interacting optimizations. The effectiveness of the DMPC depends on
%      the design of local objectives, constraints, prediction accuracy of
%      neighboring vehicles, and the stability of the interactions.
%    - 隊列の集合的な挙動は、これらの局所的で相互作用する最適化から生じます。
%      DMPCの有効性は、局所的な目的、制約、隣接車両の予測精度、および
%      相互作用の安定性の設計に依存します。
%
% LIMITATIONS OF THIS CONCEPTUAL SCRIPT: (この概念的スクリプトの制限事項)
%    - Preceding Truck Prediction: The mechanism for followers to get
%      accurate predictions of the leader's future actions is highly
%      simplified. Real DMPC systems use more robust prediction/communication.
%    - 先行トラック予測：フォロワーがリーダーの将来の行動の正確な予測を得るための
%      メカニズムは大幅に簡略化されています。実際のDMPCシステムでは、より堅牢な
%      予測/通信が使用されます。
%    - Iteration/Convergence: This script uses a single-pass sequential
%      solution within each time step. More advanced DMPC might involve
%      multiple iterations between trucks at a single time step to converge
%      to a better-coordinated plan (e.g., using game theory concepts or
%      algorithms like ADMM adapted for dynamic systems).
%    - 反復/収束：このスクリプトは、各時間ステップ内でシングルパスの逐次解法を
%      使用します。より高度なDMPCでは、より調整された計画に収束するために、
%      単一の時間ステップでトラック間で複数回の反復を行う場合があります
%      （例：ゲーム理論の概念や動的システムに適合したADMMなどのアルゴリズムを使用）。
%    - Stability: Formal stability analysis of DMPC schemes is complex.
%    - 安定性：DMPCスキームの形式的な安定性解析は複雑です。
%    - `mpcActiveSetSolver` is not directly used here because each local problem
%      is solved as an NLP with `fmincon`. To use an active-set QP solver,
%      the local problems would need to be formulated as Quadratic Programs.
%    - `mpcActiveSetSolver`は、各局所問題が`fmincon`でNLPとして解かれるため、
%      ここでは直接使用されません。アクティブセットQPソルバーを使用するには、
%      局所問題を二次計画問題として定式化する必要があります。
%
% CURRENT IMPLEMENTATION: Decentralized Model Predictive Control with
% sequential solving and simplified inter-agent prediction.
% (現在の実装：逐次解法と簡略化されたエージェント間予測を伴う分散型モデル予測制御)
% --------------------------------------------------------------------------


clear; clc; close all;  % ワークスペース、コマンドウィンドウ、図を初期化
rng(1);                 % 再現性のある結果を得るための乱数シード固定

% === シミュレーションパラメータ ===
T_simulation_sec = 60; % 総シミュレーション時間（秒）
dt = 1;              % コントローラの更新間隔（秒） (dt_control から dt に変更)
num_sim_steps = T_simulation_sec / dt;  % シミュレーションステップ数

% === 車両・プラトーンの物理パラメータ ===
N = 3;                   % トラックの台数
L_truck = 18;            % 各トラックの全長 [m]
m = 20000*ones(N,1);     % 各トラックの質量 [kg]
C_d_truck = 0.6;         % 空気抵抗係数
A_f = 10;                % 投影面積 [m^2]
C_r = 0.007;             % 転がり抵抗係数
rho_air = 1.225;         % 空気密度 [kg/m^3]
g = 9.81;                % 重力加速度 [m/s^2]
eta_drivetrain = 0.85;   % 駆動系の効率
P_idle = 5000;           % アイドル時のエンジン出力 [W]
k_fuel_g_per_J = 5.83e-5;% 燃料消費係数 [g/J]

% トラック同士のシールド効果（前方車の後ろにいることで空気抵抗が軽減される）
shielding_factors = zeros(N,1); % シールド効果係数の初期化
if N > 1, shielding_factors(2) = 0.3; end  % 2台目の空力軽減係数
if N > 2, shielding_factors(3) = 0.4; end  % 3台目
C_d_eff = C_d_truck * (1 - shielding_factors); % 有効空力抵抗係数

% === 制約条件 ===
v_min = 10; v_max = 25; % 速度の下限・上限 [m/s]
a_min = -2.0; a_max = 1.0; % 加速度の下限・上限 [m/s^2]
s_min_gap_const = 5;   % 車間距離の最小定数成分 [m] (バンパー間)
t_h = 1.0;             % 時間的ヘッドウェイ [s] (速度依存の車間距離成分用)

% === MPC予測ホライズン ===
Hp = 10;  % 予測ホライズンのステップ数（ここでは10秒先まで）

% === 重み係数の設定 ===
w_v_ref_leader = 1.0;       % リーダーが目標速度を追従するための重み
w_v_track_follower = 0.8;   % フォロワーが前方車の速度を追従する重み
w_gap_control = 1.0;        % 車間距離を一定に保つための重み
w_fuel_leader = 0.01;       % 燃費コスト（リーダー用）
w_fuel_follower = 0.05;     % 燃費コスト（フォロワー用）
w_accel_smooth = 0.01;      % 加速度の変動（ジャーク）を滑らかにするための重み

% === リーダー用の目標速度プロファイル ===
v_ref_profile = ones(num_sim_steps + Hp, 1) * 20; % 初期値は一定速度 (20 m/s)

% 速度プロファイルを三角形状に変更（加速→定速→減速）
if num_sim_steps + Hp >= 20 % プロファイル作成に必要なステップ数があるか確認
    v_ref_profile(1:10) = linspace(15, 22, 10);                         % 加速区間 (15m/s から 22m/s へ)
    v_ref_profile(11:num_sim_steps+Hp-10) = 22;                        % 定速区間 (22m/s)
    v_ref_profile(num_sim_steps+Hp-9:num_sim_steps+Hp) = linspace(22, 18, 10); % 減速区間 (22m/s から 18m/s へ)
end
v_ref_profile = min(max(v_ref_profile, v_min), v_max);  % 速度制限を適用

% === 初期状態の定義（位置・速度）===
s_platoon = zeros(N, num_sim_steps + 1);  % 各トラックの位置履歴
v_platoon = zeros(N, num_sim_steps + 1);  % 各トラックの速度履歴
a_platoon_actual = zeros(N, num_sim_steps); % 実際に適用された加速度の履歴

v_initial = v_ref_profile(1) * ones(N,1);   % 初期速度は目標プロファイルの初期値と同じ
s_initial = zeros(N,1); % 初期位置ベクトル
s_initial(1) = 0; % リーダーの初期位置は0
for i = 2:N % フォロワーの初期位置設定
    % フォロワーの初期位置は前車両からの目標間隔分だけ後ろに配置
    desired_gap_initial = L_truck + s_min_gap_const + t_h * v_initial(i); % 目標車頭間隔
    s_initial(i) = s_initial(i-1) - desired_gap_initial; % 車頭位置
end
s_platoon(:,1) = s_initial; % 初期位置を履歴に格納
v_platoon(:,1) = v_initial; % 初期速度を履歴に格納

% トラックの予測情報を初期化（主にフォロワーが前方車両の情報を格納するために使用）
predicted_s_ahead = zeros(Hp+1, 1);  % 前方車両の位置予測 (Hp+1 ステップ分)
predicted_v_ahead = zeros(Hp+1, 1);  % 前方車両の速度予測 (Hp+1 ステップ分)

fprintf('Starting Decentralized Distributed MPC Simulation...\n'); % シミュレーション開始メッセージ
for k_sim = 1:num_sim_steps  % シミュレーションの各タイムステップ (k_sim)

    if mod(k_sim,10) == 0
        fprintf('Simulation Step %d/%d\n', k_sim, num_sim_steps); % 現在のステップ表示
    end

    % 現在の各車両の位置・速度を取得
    current_s = s_platoon(:, k_sim);
    current_v = v_platoon(:, k_sim);

    % 今ステップで計算された各トラックの加速度を格納する配列（初期化）
    optimal_a_this_step = zeros(N,1);

    % =======================
    % 各トラックのMPCを個別に解く（先頭から順に、逐次的に）
    % =======================
    for i_truck = 1:N % 各トラックについてループ (i_truck = 1, 2, ..., N)
        % 1. リーダー(i_truck == 1)の場合、目標速度プロファイルを参照
        v_ref_horizon = []; % 初期化
        if i_truck == 1
            v_ref_horizon = v_ref_profile(k_sim : k_sim + Hp -1); % 現在時刻からHpステップ先までの目標速度
        end

        % 2. 直前ステップの加速度を取得（加速度変化の平滑化コスト計算用）
        a_prev_applied_this_truck = []; % デフォルトは空（初期ステップ k_sim=1 では存在しないため）
        if k_sim > 1
            a_prev_applied_this_truck = a_platoon_actual(i_truck, k_sim-1); % 前ステップで実際に適用された加速度
        end

        % 3. フォロワー(i_truck > 1)の場合、前方車両の予測を取得
        %    (このスクリプトでは簡略化された予測方法を使用)
        if i_truck > 1
            if k_sim == 1 % シミュレーションの最初のステップの場合
                % 初回ステップ：前方車両は現在の速度で等速直線運動すると仮定して予測
                predicted_s_ahead(1) = current_s(i_truck-1); % 前方車両の現在位置
                predicted_v_ahead(1) = current_v(i_truck-1); % 前方車両の現在速度
                temp_a_ahead = 0; % 前方車両の加速度を0と仮定
                for h=1:Hp % Hpステップ先まで予測
                    predicted_v_ahead(h+1) = predicted_v_ahead(h) + temp_a_ahead * dt;
                    predicted_v_ahead(h+1) = max(v_min, min(v_max, predicted_v_ahead(h+1))); % 速度制約
                    predicted_s_ahead(h+1) = predicted_s_ahead(h) + predicted_v_ahead(h)*dt + 0.5*temp_a_ahead*dt^2;
                end
            else % 2ステップ目以降
                % 前方車両が直前のステップで適用した加速度を継続すると仮定して予測
                % (より高度なDMPCでは、前方車両が計画した加速度シーケンスを使用)
                last_applied_a_ahead = 0; % デフォルトは0
                if k_sim > 1 % k_sim=1では前方車両のa_platoon_actualはまだない
                    last_applied_a_ahead = a_platoon_actual(i_truck-1, k_sim-1);
                end
                predicted_s_ahead(1) = current_s(i_truck-1);
                predicted_v_ahead(1) = current_v(i_truck-1);
                for h=1:Hp
                    predicted_v_ahead(h+1) = predicted_v_ahead(h) + last_applied_a_ahead * dt;
                    predicted_v_ahead(h+1) = max(v_min, min(v_max, predicted_v_ahead(h+1)));
                    predicted_s_ahead(h+1) = predicted_s_ahead(h) + predicted_v_ahead(h)*dt + 0.5*last_applied_a_ahead*dt^2;
                end
            end
        end

        % 4. 各トラックのMPC問題を定義
        %    (目的関数と制約条件を設定し、fminconで解く)

        % 最適化変数: トラックiの加速度シーケンス [a_i(k), ..., a_i(k+Hp-1)]
        num_dec_vars_local = Hp; % 決定変数の数 = 予測ホライズン長
        lb_a_local = ones(num_dec_vars_local, 1) * a_min;  % 加速度の下限値
        ub_a_local = ones(num_dec_vars_local, 1) * a_max;  % 加速度の上限値
        a_guess_local = zeros(num_dec_vars_local, 1);      % 加速度の初期推定値 (0 m/s^2)
        if k_sim > 1 % 2ステップ目以降は前回の加速度を初期推定値とする
            a_guess_local(:) = a_platoon_actual(i_truck, k_sim-1);
        end

        % 現在のトラックの状態と物理パラメータを取得 (objFunLocal, nonlconFunLocalへ渡すため)
        s_i_current_val = current_s(i_truck);
        v_i_current_val = current_v(i_truck);
        m_i_val = m(i_truck);
        C_d_eff_i_val = C_d_eff(i_truck);

        % 目的関数ハンドルを作成
        objFunLocal = @(a_i_seq) mpcObjectiveTruck_i(a_i_seq, i_truck, N, Hp, dt, ...
            s_i_current_val, v_i_current_val, ...
            v_ref_horizon, predicted_s_ahead, predicted_v_ahead, ...
            m_i_val, C_d_eff_i_val, A_f, C_r, rho_air, g, ...
            eta_drivetrain, P_idle, k_fuel_g_per_J, L_truck, s_min_gap_const, t_h, ...
            a_prev_applied_this_truck, ...
            w_v_ref_leader, w_v_track_follower, w_gap_control, ...
            w_fuel_leader, w_fuel_follower, w_accel_smooth);

        % 非線形制約関数ハンドルを作成（速度制限、車間距離制約）
        nonlconFunLocal = @(a_i_seq) mpcConstraintsTruck_i(a_i_seq, i_truck, N, Hp, dt, ...
            s_i_current_val, v_i_current_val, ...
            predicted_s_ahead, ... % 制約には前方車両の位置予測のみ必要
            v_min, v_max, L_truck, s_min_gap_const, t_h);

        % fminconの最適化オプション設定
        options_local_mpc = optimoptions('fmincon', 'Algorithm','sqp', ... % SQPアルゴリズムを使用
            'Display','none', 'MaxIterations', 50, ...                     % 最大反復回数
            'MaxFunctionEvaluations', 5000, 'ConstraintTolerance', 1e-3, ... % 制約許容誤差
            'OptimalityTolerance', 1e-3, 'StepTolerance', 1e-7);          % 最適性許容誤差など

        % fmincon を用いて局所最適化問題を解く（トラック i の加速度シーケンスを求める）
        [a_i_optimal_sequence, ~] = fmincon(objFunLocal, a_guess_local, ... % J_local_opt はここでは未使用
            [], [], [], [], lb_a_local, ub_a_local, ...
            nonlconFunLocal, options_local_mpc);

        % MPCの再ceding horizon原則に基づき、最適化された加速度シーケンスの最初の制御入力のみを適用
        optimal_a_this_step(i_truck) = a_i_optimal_sequence(1);

        % (高度なDMPCの場合、ここでトラックiの計画軌道を計算し、
        %  次のトラックi+1がMPCを解く際に利用できるように情報を更新・伝達する処理が入る)
    end % 各トラックのMPC計算ループ終了 (i_truck)

    % 各トラックの最適化された加速度を適用し、プラトーン全体の状態を更新
    a_platoon_actual(:, k_sim) = optimal_a_this_step; % 適用した加速度を記録

    % 速度更新（v(k+1) = v(k) + a*dt）
    v_platoon(:, k_sim+1) = v_platoon(:, k_sim) + optimal_a_this_step * dt;
    v_platoon(:, k_sim+1) = max(v_min, min(v_max, v_platoon(:, k_sim+1))); % 速度制約を厳密に適用

    % 位置更新（s(k+1) = s(k) + v*dt + 0.5*a*dt^2）
    s_platoon(:, k_sim+1) = s_platoon(:, k_sim) + v_platoon(:, k_sim) * dt ...
        + 0.5 * optimal_a_this_step * dt^2;
end % シミュレーションステップループ終了 (k_sim)
fprintf('Decentralized Distributed MPC Simulation Finished.\n'); % シミュレーション終了メッセージ

% === プロットの準備 ===
time_axis_sim = (0:num_sim_steps) * dt; % シミュレーション時間軸 (状態量プロット用)
time_axis_accel_sim = (0:num_sim_steps-1) * dt; % 加速度プロット用の時間軸

% --- 各トラックの速度プロット ---
figure; % 新しい図ウィンドウ
plot(time_axis_sim, v_platoon', 'LineWidth', 1.5); hold on; % 各トラックの速度
plot(time_axis_accel_sim, v_ref_profile(1:num_sim_steps), 'k--', 'LineWidth', 1.5); % リーダーの目標速度
title('Truck Velocities (Decentralized MPC)'); % タイトル
xlabel('Time (s)'); ylabel('Velocity (m/s)'); % 軸ラベル
legend_entries_v = cell(N+1,1); % 凡例用セル配列
for i=1:N % 各トラックの凡例
    legend_entries_v{i} = sprintf('Truck %d', i);
end
legend_entries_v{N+1} = 'Reference (Leader)'; % リーダー目標速度の凡例
legend(legend_entries_v, 'Location', 'best'); grid on; % 凡例表示とグリッド

% --- 各トラック間の車間距離プロット ---
if N > 1 % トラックが2台以上の場合のみプロット
    figure; % 新しい図ウィンドウ
    for i = 2:N % 各フォロワーについて (i=2 から N まで)
        actual_gap = (s_platoon(i-1,:) - L_truck) - s_platoon(i,:); % 実際のバンパー間距離
        desired_min_gap = s_min_gap_const + t_h * v_platoon(i,:);   % 望ましい最小バンパー間ギャップ (フォロワー自身の速度に依存)
        subplot(N-1,1,i-1); % サブプロット作成
        plot(time_axis_sim, actual_gap, 'b-', 'LineWidth', 1.2); hold on; % 実際のギャップ
        plot(time_axis_sim, desired_min_gap, 'r--', 'LineWidth', 1.2);   % 望ましい最小ギャップ
        title(sprintf('Gap T%d-T%d (DMPC)', i-1,i)); ylabel('Gap (m)'); grid on; % タイトルとラベル
        legend('Actual Bumper-Gap', 'Desired Min Bumper-Gap', 'Location', 'best'); % 凡例
    end
    sgtitle('Inter-truck Gaps (Decentralized MPC)'); % 図全体のタイトル
end
%%  MPC Objective Function for Truck i (トラックiのためのMPC目的関数)
function J_i = mpcObjectiveTruck_i(a_i_seq, i_truck, N_total_trucks, Hp, dt, ...
    s_i_current, v_i_current, ...
    v_ref_leader_horizon, s_ahead_pred, v_ahead_pred, ...
    m_i, C_d_eff_i, A_f, C_r, rho_air, g, ...
    eta_drivetrain, P_idle, k_fuel_g_per_J, L_truck, s_min_gap_const_i, t_h_i, ...
    a_prev_applied_i, ... % 直前のステップで適用された加速度 (平滑化用)
    w_v_ref_L, w_v_track_F, w_gap_F, w_fuel_L, w_fuel_F, w_accel_smooth_i) %#ok<INUSD> % 未使用引数の警告抑制

% トラックiの加速度シーケンス a_i_seq に基づいて、予測ホライズンHpステップ先までの状態を予測
s_i_pred = zeros(Hp+1, 1); % 位置予測
v_i_pred = zeros(Hp+1, 1); % 速度予測
s_i_pred(1) = s_i_current; % 初期位置
v_i_pred(1) = v_i_current; % 初期速度
fuel_i_total = 0; % トラックiの総燃料消費量 (このホライズン内)

for h = 1:Hp % 予測ホライズンの各ステップで計算
    % ダイナミクス (運動方程式)
    v_i_pred(h+1) = v_i_pred(h) + a_i_seq(h) * dt; % 速度更新
    s_i_pred(h+1) = s_i_pred(h) + v_i_pred(h) * dt + 0.5 * a_i_seq(h) * dt^2; % 位置更新

    % トラックiのステップhにおける燃料消費量計算
    current_v_fuel = v_i_pred(h); % 区間開始時の速度を使用
    if current_v_fuel < 0.1, current_v_fuel = 0.1; end % ゼロ速度付近での計算エラー回避
    F_aero = 0.5 * rho_air * C_d_eff_i * A_f * current_v_fuel^2; % 空気抵抗
    F_roll = m_i * g * C_r; % 転がり抵抗
    F_inertia = m_i * a_i_seq(h); % 慣性抵抗
    F_total = F_aero + F_roll + F_inertia; % 総走行抵抗
    P_wheel = F_total * current_v_fuel; % 車輪出力
    if P_wheel >= 0 % 正の駆動力の場合
        P_engine = P_wheel / eta_drivetrain + P_idle; % エンジン出力
    else % 負の駆動力（制動）の場合
        P_engine = P_idle; % アイドル状態と仮定
    end
    fuel_i_total = fuel_i_total + (k_fuel_g_per_J * P_engine * dt); % 燃料消費量加算
end

% コスト要素の計算
cost_v_tracking = 0;    % 速度追従コスト
cost_gap_control = 0;   % 車間距離制御コスト
cost_fuel = 0;          % 燃料消費コスト

%  加速度平滑化コスト
% Option 1: 加速度の2乗和をペナルティとする (単純で効果的)
% cost_accel_squared = w_accel_smooth_i * sum(a_i_seq.^2);

% Option 2: 予測ホライズン内での加速度変化（ジャーク）をペナルティとする
cost_accel_diff_horizon = 0;
if length(a_i_seq) > 1
    cost_accel_diff_horizon = w_accel_smooth_i * sum(diff(a_i_seq).^2);
end

% Option 3: 直前の実加速度から計画された最初の加速度への変化をペナルティとする
cost_accel_diff_initial_step = 0;
if ~isempty(a_prev_applied_i) % 直前の加速度が利用可能な場合
    cost_accel_diff_initial_step = w_accel_smooth_i * (a_i_seq(1) - a_prev_applied_i)^2;
else % 初期ステップ、または直前の加速度が提供されない場合
    cost_accel_diff_initial_step = w_accel_smooth_i * (a_i_seq(1) - 0)^2; % 最初の加速度を0からの変化とみなす
end
% 平滑化コストの選択（ここではOption2とOption3の組み合わせ）
cost_accel_smooth = cost_accel_diff_horizon + cost_accel_diff_initial_step;


if i_truck == 1 % リーダーの場合
    cost_v_tracking = w_v_ref_L * sum((v_i_pred(2:end) - v_ref_leader_horizon(1:Hp)).^2); % 目標速度追従コスト
    cost_fuel = w_fuel_L * fuel_i_total; % 燃料コスト
    cost_gap_control = 0; % リーダーには車間制御コストなし
else % フォロワーの場合
    % 速度追従コスト (前方車両の予測速度に追従)
    cost_v_tracking = w_v_track_F * sum((v_i_pred(2:end) - v_ahead_pred(2:end)).^2);

    % 車間距離制御コスト (バンパー間距離)
    actual_bumper_gap = (s_ahead_pred(2:end) - L_truck) - s_i_pred(2:end); % 実際のバンパー間距離
    target_bumper_gap = s_min_gap_const_i + t_h_i * v_i_pred(2:end); % 目標バンパー間距離 (自身の速度に依存)
    cost_gap_control = w_gap_F * sum((actual_bumper_gap - target_bumper_gap).^2);

    cost_fuel = w_fuel_F * fuel_i_total; % 燃料コスト
end

J_i = cost_v_tracking + cost_gap_control + cost_fuel + cost_accel_smooth; % 総コスト
end

%%  MPC Constraints Function for Truck i (トラックiのためのMPC制約関数)
function [c, ceq] = mpcConstraintsTruck_i(a_i_seq, i_truck, N_total_trucks, Hp, dt, ...
    s_i_current, v_i_current, ...
    s_ahead_pred, ... % 前方車両の予測位置 (制約用)
    v_min_i, v_max_i, L_truck_i, s_min_gap_const_i, t_h_i) %#ok<INUSD> % 未使用引数の警告抑制

% トラックiの状態を予測
s_i_pred = zeros(Hp+1, 1);
v_i_pred = zeros(Hp+1, 1);
s_i_pred(1) = s_i_current;
v_i_pred(1) = v_i_current;

for h = 1:Hp
    v_i_pred(h+1) = v_i_pred(h) + a_i_seq(h) * dt;
    s_i_pred(h+1) = s_i_pred(h) + v_i_pred(h) * dt + 0.5 * a_i_seq(h) * dt^2;
end

c = []; % 不等式制約 c(x) <= 0
% トラックiの速度制約 (ステップ 1 から Hp まで、すなわち v_i_pred(2:end))
c = [c; v_i_pred(2:end) - v_max_i]; % v_i <= v_max  => v_i - v_max <= 0
c = [c; v_min_i - v_i_pred(2:end)]; % v_i >= v_min  => v_min - v_i <= 0

% フォロワーの場合の車間距離制約 (安全制約)
if i_truck > 1
    % s_ahead_pred(h+1) - s_i_pred(h+1) >= L_truck_i + s_min_gap_const_i + t_h_i * v_i_pred(h+1)  (車頭間距離)
    % これを変形して c(x) <= 0 の形にする:
    % (L_truck_i + s_min_gap_const_i + t_h_i * v_i_pred(h+1)) - (s_ahead_pred(h+1) - s_i_pred(h+1)) <= 0
    desired_front_to_front_sep = L_truck_i + s_min_gap_const_i + t_h_i * v_i_pred(2:end); % 目標最小車頭間隔
    actual_front_to_front_sep = s_ahead_pred(2:end) - s_i_pred(2:end); % 実際の車頭間隔
    c_spacing = desired_front_to_front_sep - actual_front_to_front_sep; % 制約違反量 (正だと違反)
    c = [c; c_spacing]; % 制約リストに追加
end
ceq = []; % 等式制約 ceq(x) = 0 (今回はなし)
end