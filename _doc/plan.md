# NN4JS 架構規劃

## 目標
將 `ref/nn2kv/` 的 Python/NumPy 深度學習框架翻譯成 JavaScript 版本，使用 `numjs` 作為數值運算庫。

## 檔案結構

```
src/
├── tensor.js    # 張量類別與自動微分
├── nn.js        # 網路層與優化器
├── gpt.js       # GPT 模型與注意力機制
├── chargpt.js   # 訓練與推論函数
└── main.js      # 主程式入口
```

## 模組設計

### 1. tensor.js
- `Tensor` 類別：封裝 numjs 陣列，支援自動微分
  - 建構子：`new Tensor(data, _children?, requires_grad?)`
  - 反向傳播：`backward()`
  - 梯度歸零：`zero_grad()`
- 運算子：
  - `__add__` / `__mul__` / `__matmul__` (矩陣乘法)
  - `__sub__` / `__truediv__` / `__pow__`
- 張量操作：
  - `transpose(ax1, ax2)` - 維度交換
  - `reshape(...shape)` - 形狀重塑
- 激活函數：
  - `relu()` - ReLU 激活
  - `softmax(axis)` - Softmax
  - `cross_entropy(targets)` - 交叉熵損失
  - `masked_fill(mask, value)` - 遮罩填充
- 工具函式：
  - `unbroadcast(grad, shape)` - 處理廣播梯度
  - `cat(tensors, axis)` - 張量拼接

### 2. nn.js
- `Module` 基底類別：遞迴收集參數
  - `parameters()` - 取得所有 requires_grad 的張量
- 網路層：
  - `Linear(in_features, out_features, bias)` - 全連接層
  - `Embedding(num_embeddings, embedding_dim)` - 詞嵌入層
  - `RMSNorm(dim, eps)` - RMS 正規化
- 優化器：
  - `Adam(params, lr, betas, eps)` - Adam 優化器

### 3. gpt.js
- `CausalSelfAttention(n_embd, n_head)` - 因果自注意力 (KV Cache 支援)
- `MLP(n_embd)` - 前饋網路
- `Block(n_embd, n_head)` - Transformer 區塊
- `GPT(vocab_size, block_size, n_layer, n_embd, n_head)` - GPT 模型

### 4. chargpt.js
- `train_model(model, optimizer, docs, uchars, BOS, block_size, num_steps)` - 訓練函式
- `generate_samples(model, uchars, BOS, vocab_size, block_size, num_samples, temperature)` - 推論生成

### 5. main.js
- 資料讀取與前處理
- 模型初始化
- 訓練與推論流程

## numjs 使用對照

| Python NumPy | numjs |
|-------------|-------|
| `np.array()` | `nj.array()` |
| `np.zeros_like()` | `nj.zeros_like()` |
| `np.sum(axis)` | `.sum(axis)` |
| `np.max(axis, keepdims)` | `.max(axis)` (需另行處理 keepdims) |
| `np.exp()` | `nj.exp()` |
| `@` (矩陣乘法) | `.dot()` |
| `.T` | `.T` |
| `np.swapaxes()` | 需用 reshape + transpose |

## 注意事項
- numjs 缺乏完整 NumPy 功能，部分操作需繞過實現
- 自動微分需完整模擬 Python 版本的計算圖机制
- KV Cache 在推論時的處理與 Python 版本相同

## 依賴
- numjs: `npm install numjs`