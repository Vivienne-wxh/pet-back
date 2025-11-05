# Git 代理配置指南

## 常见 VPN 代理端口

- Clash: 7890 (HTTP), 7891 (SOCKS5)
- V2Ray: 1080 (SOCKS5)
- Shadowsocks: 1080 (SOCKS5)
- 其他: 检查 VPN 设置中的本地代理端口

## 配置 Git 使用代理

### 方法1：HTTP 代理（推荐）
```bash
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

### 方法2：SOCKS5 代理
```bash
git config --global http.proxy socks5://127.0.0.1:1080
git config --global https.proxy socks5://127.0.0.1:1080
```

### 查看当前配置
```bash
git config --global --get http.proxy
git config --global --get https.proxy
```

### 取消代理
```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

