# Kimina Lean Server

P0 使用固定 Docker image：`projectnumina/kimina-lean-server:2.0.0`。

## 启动

在仓库根目录执行：

```bash
cp .env.example .env
docker compose -f infra/lean-server/compose.yaml up -d
docker compose -f infra/lean-server/compose.yaml logs -f
```

服务默认监听 `http://127.0.0.1:8000`。基础 API 测试：

```bash
curl --request POST \
  --url http://127.0.0.1:8000/verify \
  --header 'Content-Type: application/json' \
  --data '{"codes":[{"custom_id":"positive","proof":"#check Nat"}],"infotree_type":"original"}'
```

项目代码中的 `tinylean_rl.verifier.kimina.verify_code` 使用同一个 `/verify` endpoint。P0 不在宿主机安装 Lean；如需更换 Lean/mathlib 版本，必须先更新复现记录并重新确认方案。

