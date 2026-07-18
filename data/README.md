# 原始 mock 数据

此目录是本项目唯一的企业事实数据源。本地应用、测试和 Docker 镜像都直接读取以下
三份用户提供的原始文件：

- `person 1.json`
- `company 1.json`
- `relations 1.json`

`backend/app/tools/repository.py` 会验证原始数组，并仅在内存中投影 typed graph
records。目录中不保留 aliases、manifest、evidence 等旧生成 JSON；早期的
`curated relations.json` 也已删除，系统不会补造控制关系。
