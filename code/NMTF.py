from statistics import stdev
import torch
from scipy.special import xlogy
from tensorly import check_random_state
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error
import torch
import time

def kl_divergence(y, y_hat):
    epsilon = 1e-10
    return y * torch.log((y + epsilon) / (y_hat + epsilon)) + (1 - y) * torch.log((1 - y + epsilon) / (1 - y_hat + epsilon))

# 更新式の関数
def update_ty(Ty, Uy, Vy, U2, V2, y, y_hat, a2, a2_hat, eta2, learning_rate):
    # 分母の計算
    denominator_y = torch.einsum('jr,kr->r', Uy, Vy)
    denominator_a2 = torch.einsum('jr,kr->r', U2, V2)
    denominator = denominator_y + eta2 * denominator_a2
    # 分子の第一項の計算: torch.where を使ってyが0でない要素だけ計算
    masked_y = torch.where(y != 0, y / y_hat, torch.zeros_like(y))
    numerator_y = torch.einsum('ijk,jr,kr->ir', masked_y, Uy, Vy)
    masked_a2 = torch.where(a2 != 0, a2 / a2_hat, torch.zeros_like(a2))
    numerator_a2 = torch.einsum('ijk,jr,kr->ir', masked_a2, U2, V2)
    # 分子の計算
    numerator = numerator_y + eta2 * numerator_a2
    # 更新式の計算
    delta_Ty = Ty * numerator / denominator.unsqueeze(0)
    new_Ty = Ty + learning_rate * (delta_Ty - Ty)
    return new_Ty

def update_uy(Ty, Uy, Vy, T1, V1, y, y_hat, a1, a1_hat, eta1, learning_rate):
    denominator_y = torch.einsum('ir,kr->r', Ty, Vy)
    denominator_a1 = torch.einsum('ir,kr->r', T1, V1)
    denominator = denominator_y + eta1 * denominator_a1
    masked_y = torch.where(y != 0, y / y_hat, torch.zeros_like(y))
    numerator_y = torch.einsum('ijk,ir,kr->jr', masked_y, Ty, Vy)
    masked_a1 = torch.where(a1 != 0, a1 / a1_hat, torch.zeros_like(a1))
    numerator_a1 = torch.einsum('ijk,ir,kr->jr', masked_a1, T1, V1)
    numerator = numerator_y + eta1 * numerator_a1
    delta_Uy = Uy * numerator / denominator.unsqueeze(0)  
    new_Uy = Uy + learning_rate * (delta_Uy - Uy)
    return new_Uy

def update_vy(Ty, Uy, Vy, y, y_hat, learning_rate):
    denominator = torch.einsum('ir,jr->r', Ty, Uy)
    masked_y = torch.where(y != 0, y / y_hat, torch.zeros_like(y))
    numerator = torch.einsum('ijk,ir,jr->kr', masked_y, Ty, Uy)
    delta_Vy = Vy * numerator / denominator.unsqueeze(0)
    new_Vy = Vy + learning_rate * (delta_Vy - Vy)
    return new_Vy

def update_t1(T1, Uy, V1, a1, a1_hat, learning_rate):
    masked_a1 = torch.where(a1 != 0, a1 / a1_hat, torch.zeros_like(a1))
    numerator = torch.einsum('ijk,jr,kr->ir', masked_a1, Uy, V1)
    denominator = torch.einsum('jr,kr->r', Uy, V1)
    delta_T1 = T1 * numerator / denominator.unsqueeze(0)
    new_T1 = T1 + learning_rate * (delta_T1 - T1)
    return new_T1

def update_v1(T1, Uy, V1, a1, a1_hat, learning_rate):
    masked_a1 = torch.where(a1 != 0, a1 / a1_hat, torch.zeros_like(a1))
    numerator = torch.einsum('ijk,ir,jr->kr', masked_a1, T1, Uy)
    denominator = torch.einsum('ir,jr->r', T1, Uy)
    delta_V1 = V1 * numerator / denominator.unsqueeze(0)
    new_V1 = V1 + learning_rate * (delta_V1 - V1)
    return new_V1

def update_v2(Ty, U2, V2, a2, a2_hat, learning_rate):
    masked_a2 = torch.where(a2 != 0, a2 / a2_hat, torch.zeros_like(a2))
    numerator = torch.einsum('ijk,ir,jr->kr', masked_a2, Ty, U2)
    denominator = torch.einsum('ir,jr->r', Ty, U2)
    delta_V2 = V2 * numerator / denominator.unsqueeze(0)
    new_V2 = V2 + learning_rate * (delta_V2 - V2)
    return new_V2

def update_u2(Ty, U2, V2, a2, a2_hat, learning_rate):
    masked_a2 = torch.where(a2 != 0, a2 / a2_hat, torch.zeros_like(a2))
    numerator = torch.einsum('ijk,ir,kr->jr', masked_a2, Ty, V2)
    denominator = torch.einsum('ir,kr->r', Ty, V2)
    delta_U2 = U2 * numerator / denominator.unsqueeze(0)
    new_U2 = U2 + learning_rate * (delta_U2 - U2)
    return new_U2


# トレーニング関数
def train(Ty, Uy, Vy, T1, V1, U2, V2, y, a1, a2, eta1, eta2, learning_rate, tolerance=1e-4, max_iterations=1000):
    # テンソルに変換
    Ty = torch.tensor(Ty, dtype=torch.float32, requires_grad=False)
    Uy = torch.tensor(Uy, dtype=torch.float32, requires_grad=False)
    Vy = torch.tensor(Vy, dtype=torch.float32, requires_grad=False)
    T1 = torch.tensor(T1, dtype=torch.float32, requires_grad=False)
    V1 = torch.tensor(V1, dtype=torch.float32, requires_grad=False)
    U2 = torch.tensor(U2, dtype=torch.float32, requires_grad=False)
    V2 = torch.tensor(V2, dtype=torch.float32, requires_grad=False)
    y = torch.tensor(y, dtype=torch.float32, requires_grad=False)
    a1 = torch.tensor(a1, dtype=torch.float32, requires_grad=False)
    a2 = torch.tensor(a2, dtype=torch.float32, requires_grad=False)

    for iteration in range(max_iterations):
        # 推定値の計算
        y_hat = torch.einsum('ir,jr,kr->ijk', Ty, Uy, Vy)
        a1_hat = torch.einsum('ir,jr,kr->ijk', T1, Uy, V1)
        a2_hat = torch.einsum('ir,jr,kr->ijk', Ty, U2, V2)

        # 損失関数の計算
        loss = torch.sum(kl_divergence(y, y_hat)) + eta1 * torch.sum(kl_divergence(a1, a1_hat)) + eta2 * torch.sum(kl_divergence(a2, a2_hat))
        
        if loss < tolerance:
            break

        # 更新
        #print(y_hat)
        Ty = update_ty(Ty, Uy, Vy,  U2, V2, y, y_hat, a2, a2_hat, eta2,learning_rate)
        Uy = update_uy(Ty, Uy, Vy, T1, V1, y, y_hat, a1, a1_hat, eta1,learning_rate)
        Vy = update_vy(Ty, Uy, Vy, y, y_hat,learning_rate)
        T1 = update_t1(T1, Uy, V1, a1, a1_hat,learning_rate)
        V1 = update_v1(T1, Uy, V1, a1, a1_hat,learning_rate)
        V2 = update_v2(Ty, U2, V2, a2, a2_hat,learning_rate)
        U2 = update_u2(Ty, U2, V2, a2, a2_hat,learning_rate)
    
    return Ty.numpy(), Uy.numpy(), Vy.numpy(), T1.numpy(), V1.numpy(), U2.numpy(), V2.numpy()

def generate_nonnegative_matrices_NMTF(Iy, Jy, Ky, R, random_state=1234):
    torch.manual_seed(random_state)  # ランダムシードの設定
    while True:
        Ty = torch.rand((Iy, R))+1   # Iy x R 行列を生成 (1 から 2 の範囲で生成)
        Uy = torch.rand((Jy, R)) +1  # Jy x R 行列を生成 (1 から 2 の範囲で生成)
        Vy = torch.rand((Ky, R)) +1  # Ky x R 行列を生成 (1 から 2 の範囲で生成)

        # 条件を満たすか確認
        result = torch.einsum('ir,jr,kr->ijk', Ty, Uy, Vy)
        if (result > 0).all():
            return Ty, Uy, Vy

def calculate_rmse(original, estimate):
    return torch.sqrt(torch.mean((original - estimate) ** 2))

def calculate_rmse_for_R_values(R_values, y, a1, a2, eta1, eta2, learning_rate,iterations, A, B, C, D, E, F, G):
    rmse_results = {}
    time_elapsed = []
    Iy, Jy, Ky = y.shape
    I1, J1, K1 = a1.shape
    I2, J2, K2 = a2.shape

    # 開始時刻の記録
    start_time = time.time()
    for R in R_values:
        # 初期値の生成
        Ty, Uy, Vy = generate_nonnegative_matrices_NMTF(Iy, Jy, Ky, R)
        T1, _, V1 = generate_nonnegative_matrices_NMTF(I1, Jy, K1, R)
        _, U2, V2 = generate_nonnegative_matrices_NMTF(Iy, J2, K2, R)
        
        # トレーニングの実行
        t_y, u_y, v_y, t1, v1, u2, v2 = train(Ty, Uy, Vy, T1, V1, U2, V2, y, a1, a2, eta1, eta2, learning_rate, tolerance=0.01, max_iterations=iterations)
        
        
        # 結果のテンソルに変換
        t_y = torch.from_numpy(t_y)
        u_y = torch.from_numpy(u_y)
        v_y = torch.from_numpy(v_y)
        t1 = torch.from_numpy(t1)
        v1 = torch.from_numpy(v1)
        u2 = torch.from_numpy(u2)
        v2 = torch.from_numpy(v2)

        # 推定値の計算
        y_hat = torch.einsum('ir,jr,kr->ijk', t_y, u_y, v_y)
        a1_hat = torch.einsum('ir,jr,kr->ijk', t1, u_y, v1)
        a2_hat = torch.einsum('ir,jr,kr->ijk', t_y, u2, v2)

        # RMSEの計算
        rmse_y = calculate_rmse(y, y_hat)
        rmse_a1 = calculate_rmse(a1, a1_hat)
        rmse_a2 = calculate_rmse(a2, a2_hat)

        # 結果の保存
        rmse_results[R] = {'rmse_y': rmse_y, 'rmse_a1': rmse_a1, 'rmse_a2': rmse_a2}
        current_time = time.time()
        elapsed = current_time - start_time
        time_elapsed.append(elapsed)

    return rmse_results,time_elapsed,v_y


