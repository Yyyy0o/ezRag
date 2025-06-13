from mcp.server.fastmcp import FastMCP
from datetime import datetime

# 创建 MCP 实例
mcp = FastMCP("Demo")

# 示例工具
@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

@mcp.tool("时间查询工具")
def currentDate() -> str:
    return datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒")

# 示例资源
@mcp.resource("greeting://{name}")
def greet(name: str) -> str:
    return f"Hello, {name}!"

# 启动服务
if __name__ == "__main__":
    print("Starting MCP server...")
    mcp.run()