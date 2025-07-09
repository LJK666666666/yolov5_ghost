好的，没问题。基于我们之前讨论的策略（使用新的4分类标签，构建一个处理完整图像的Conditional VAE），我为您提供一个详细的、带有注释的伪代码，它将作为一个清晰的框架，您可以使用PyTorch或TensorFlow等框架轻松地将其转化为实际代码。

这个伪代码将涵盖从配置、数据加载到模型构建、训练和最终生成图像的全过程。

-----

### **Conditional VAE (CVAE) 构建与训练伪代码**

#### **1. 配置与超参数 (Configuration & Hyperparameters)**

```python
# --- 模型和数据配置 ---
IMG_HEIGHT = 128          # 图像统一调整后的高度
IMG_WIDTH = 128           # 图像统一调整后的宽度
CHANNELS = 3              # 图像通道数 (RGB)
INPUT_SHAPE = (CHANNELS, IMG_HEIGHT, IMG_WIDTH) # Pytorch-style input shape

# --- VAE 核心超参数 ---
LATENT_DIM = 256          # 隐空间Z的维度。这是一个关键参数，需要调试。
NUM_CLASSES = 4           # 类别数量 ("fully_equipped", "helmet_only", "vest_only", "no_equipment")

# --- 训练超参数 ---
LEARNING_RATE = 0.001     # 学习率
BATCH_SIZE = 64           # 批量大小
EPOCHS = 100              # 训练轮次
BETA = 1.0                # KL散度损失的权重 (β-VAE)。BETA > 1.0 会增强解耦性，但可能牺牲重建质量。
```

#### **2. 数据加载与预处理 (Data Loading & Preprocessing)**

```python
# 导入必要的库 (如 PyTorch, torchvision)
import framework as nn # 代表 torch.nn 或 tf.keras.layers
import dataloader_library # 代表 torch.utils.data or tf.data

# 1. 定义数据转换流程
#    - 缩放: 将图片调整到统一尺寸 (IMG_HEIGHT, IMG_WIDTH)
#    - 转换为Tensor: 将图片数据格式化
#    - 标准化: 将像素值从 [0, 255] 缩放到 [-1, 1] 或 [0, 1]，这有助于模型稳定训练
data_transforms = Compose([
    Resize((IMG_HEIGHT, IMG_WIDTH)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 2. 创建自定义数据集类
#    它需要能够返回一张处理过的图片和它对应的类别标签 (整数形式, e.g., 0-3)
class SafetyDataset(dataloader_library.Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels # 标签应为整数 [0, 1, 2, 3]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = LoadImage(self.image_paths[index]) # 从路径加载图片
        label = self.labels[index]
        image_tensor = self.transform(image)
        return image_tensor, label

# 3. 准备数据加载器 (DataLoader)
#    - `all_image_paths`: 所有图片的文件路径列表
#    - `all_labels`: 与上面路径一一对应的、已经转换成整数的标签列表
train_dataset = SafetyDataset(all_image_paths, all_labels, transform=data_transforms)
train_loader = dataloader_library.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
```

#### **3. CVAE模型架构 (CVAE Model Architecture)**

```python
# --- 编码器 (Encoder) ---
# 输入: 图像 + 条件(标签)
# 输出: 隐分布的均值(mu)和对数方差(log_var)
class Encoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Encoder, self).__init__()
        # 用于将类别标签转换为向量的嵌入层
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # 卷积层，用于从图像中提取特征
        self.conv_layers = nn.Sequential(
            # 示例结构, 实际需要根据图像大小设计
            nn.Conv2d(CHANNELS + 1, 32, kernel_size=4, stride=2, padding=1), # 输入通道为 CHANNELS+1，因为要拼接标签信息
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten() # 展平特征图
        )
        
        # 全连接层，输出mu和log_var
        self.fc_mu = nn.Linear(some_flattened_dim, latent_dim)
        self.fc_log_var = nn.Linear(some_flattened_dim, latent_dim)

    def forward(self, image, label):
        # 将标签嵌入并塑造成与图像相同的空间维度，以便拼接
        embedded_label = self.label_embedding(label)
        embedded_label = embedded_label.view(-1, 1, 1, 1).expand(-1, 1, IMG_HEIGHT, IMG_WIDTH)
        
        # 将图像和嵌入后的标签在通道维度上拼接
        conditional_input = concat([image, embedded_label], dim=1)
        
        features = self.conv_layers(conditional_input)
        mu = self.fc_mu(features)
        log_var = self.fc_log_var(features)
        return mu, log_var

# --- 解码器 (Decoder) ---
# 输入: 隐向量Z + 条件(标签)
# 输出: 重建的图像
class Decoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Decoder, self).__init__()
        # 标签嵌入层
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # 全连接层，将隐向量和标签的组合映射回卷积特征图的尺寸
        self.fc_layer = nn.Linear(latent_dim + num_classes, some_flattened_dim)

        # 转置卷积层 (或上采样+卷积)，用于从特征图重建图像
        self.deconv_layers = nn.Sequential(
            # 结构需要与编码器对称
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # 输出层激活函数，匹配[-1, 1]的标准化范围
        )

    def forward(self, z, label):
        # 将隐向量z和嵌入后的标签拼接
        embedded_label = self.label_embedding(label)
        conditional_input = concat([z, embedded_label], dim=1)
        
        features = self.fc_layer(conditional_input)
        features = features.view(-1, 128, some_height, some_width) # Reshape回特征图
        reconstructed_image = self.deconv_layers(features)
        return reconstructed_image

# --- CVAE 主模型 ---
# 将编码器和解码器组合起来
class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(ConditionalVAE, self).__init__()
        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

    # VAE核心：重参数化技巧 (Reparameterization Trick)
    # 使得我们可以从潜在分布中采样，同时保持梯度可回传
    def reparameterize(self, mu, log_var):
        std = exp(0.5 * log_var) # 计算标准差
        epsilon = sample_from_standard_normal(std.shape) # 从标准正态分布中采样噪声
        return mu + epsilon * std # z = mu + eps * sigma

    def forward(self, image, label):
        mu, log_var = self.encoder(image, label)
        z = self.reparameterize(mu, log_var)
        reconstructed_image = self.decoder(z, label)
        return reconstructed_image, mu, log_var
```

#### **4. 损失函数 (Loss Function)**

```python
def vae_loss_function(reconstructed_image, original_image, mu, log_var):
    # 1. 重建损失 (Reconstruction Loss)
    #    衡量生成图像与原始图像的相似度
    reconstruction_loss = MeanSquaredError(reconstructed_image, original_image)
    
    # 2. KL散度损失 (KL Divergence Loss)
    #    衡量编码器输出的分布与标准正态分布的差异，是一种正则化项
    #    公式: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence = -0.5 * sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # 总损失 = 重建损失 + beta * KL散度损失
    total_loss = reconstruction_loss + BETA * kl_divergence
    return total_loss
```

#### **5. 训练循环 (Training Loop)**

```python
# 1. 初始化模型和优化器
model = ConditionalVAE(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# 2. 开始训练
for epoch in range(EPOCHS):
    model.train() # 设置为训练模式
    total_train_loss = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # images: [BATCH_SIZE, CHANNELS, H, W]
        # labels: [BATCH_SIZE]
        
        # 前向传播
        reconstructed_images, mu, log_var = model(images, labels)
        
        # 计算损失
        loss = vae_loss_function(reconstructed_images, images, mu, log_var)
        
        # 反向传播和优化
        optimizer.zero_grad() # 清空梯度
        loss.backward()       # 计算梯度
        optimizer.step()      # 更新权重
        
        total_train_loss += loss.item()

    # 打印每个epoch的平均损失
    print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {total_train_loss / len(train_loader)}")

    # (可选但推荐) 在每个epoch后，进行一次评估和图像生成，以监控训练进程
    # ...
```

#### **6. 图像生成 (Inference / Generation)**

```python
def generate_images(model, num_images, desired_label_int):
    model.eval() # 设置为评估模式
    
    with no_grad(): # 关闭梯度计算
        # 1. 随机从标准正态分布中采样隐向量 z
        z = sample_from_standard_normal(shape=(num_images, LATENT_DIM))
        
        # 2. 准备你想要的类别标签
        labels = create_tensor([desired_label_int] * num_images)
        
        # 3. 使用解码器生成图像
        #    注意：在生成时，我们只需要解码器！
        generated_images = model.decoder(z, labels)
        
        # 4. 将图像反标准化（从[-1, 1]转回[0, 255]）并保存或显示
        # ...
    
    return generated_images

# --- 使用示例 ---
# 假设 "helmet_only" 对应的整数是 1
new_images = generate_images(model, num_images=10, desired_label_int=1)
SaveImages(new_images) # 保存生成的10张 "只有头盔" 的图片
```