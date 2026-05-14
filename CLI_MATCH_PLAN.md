# expflow CLI Match Plan — 对齐测量面计划架构

**目标:** 将 expflow CLI 从现有 6 个子命令组架构，重构为测量面计划中定义的扁平化顶层命令架构

## 目标 CLI 架构

```
expflow [OPTIONS] COMMAND [ARGS]

# 基础
  version              Show expflow version
  info                 Show system and environment info

# 实验管理 (来自 clearml + run)
  list [--status] [--project] [--limit]   列出实验
  show <task_id>                           实验详情
  compare <t1> <t2>                        实验对比
  stop <task_id>                           取消实验
  run <script> [args]                      提交实验 [已有]

# 超参 (来自 optuna)
  hpo <script> [--trials] [--n-jobs] [--params]    完整 HPO 流程
  hpo resume <study>
  hpo plot <study> [--type]

# 数据集
  register [--name] [--path] [--version] [--compliance]   注册数据集
  dataset list [--name] [--compliance]                    列出数据集

# 监控
  status                          组件健康检查
  board [--port]                  TensorBoard 启动器

# 观测 (来自 langfuse)
  traces [--limit] [--user-id]    列出 traces
  trace <id>                      单个 trace
  trace-cost <id>                 trace 成本
  sessions [--limit]              列出 sessions
  session <id>                    单个 session
  metrics                         聚合指标

# 配置
  init                            交互式配置
  config                          显示配置

# MCP
  mcp                             启动 MCP Server

# 旧子命令组 (向后兼容，标记 deprecated)
  clearml [...]                   (deprecated — 用 list/show/stop 代替)
  optuna [...]                    (deprecated — 用 hpo 代替)
  langfuse [...]                  (deprecated — 用 traces/trace/sessions 代替)
  run list/status/cancel          (deprecated — 用 list/status/stop 代替)
  audit [...]                     (deprecated — 用 compare/status 代替)
```

## 执行顺序

1. cli.py 重构 — 注册所有新顶层命令，保留旧组向后兼容
2. 新命令委托实现（list → clearml list_tasks, show → clearml get_task, 等）
3. 新功能模块（status.py, board.py, hpo.py, mcp.py, init.py）
4. compare 实现
5. 测试
6. AGENTS.md + PLAN.md 更新
