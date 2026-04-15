from serpapi import SerpApiClient
import os
import ast
import operator
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv


_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_expr(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅支持数字常量。")
    if isinstance(node, ast.BinOp):
        left = _safe_eval_expr(node.left)
        right = _safe_eval_expr(node.right)
        op_type = type(node.op)
        if op_type not in _ALLOWED_BIN_OPS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        return _ALLOWED_BIN_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval_expr(node.operand)
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY_OPS:
            raise ValueError(f"不支持的一元运算: {op_type.__name__}")
        return _ALLOWED_UNARY_OPS[op_type](operand)
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


def calculator(expression: str) -> str:
    """安全计算数学表达式，仅支持基础算术运算。"""
    cleaned = expression.strip().replace("×", "*").replace("÷", "/")
    if not cleaned:
        return "错误:Calculator 输入不能为空。"

    try:
        tree = ast.parse(cleaned, mode="eval")
        result = _safe_eval_expr(tree)
        return str(result)
    except ZeroDivisionError:
        return "错误:除数不能为 0。"
    except Exception as e:
        return f"错误:Calculator 无法解析表达式 '{expression}'，原因: {e}"

def search(query: str) -> str:
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"
        
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn", # 语言代码
        }
        
        client = SerpApiClient(params)
        results = client.get_dict()
        
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"
        
    except Exception as e:
        return f"搜索时发生错误: {e}"
        
class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        
    def registerTool(self, name: str, description: str, func: callable, keywords: Optional[List[str]] = None):
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        normalized_keywords = [kw.strip().lower() for kw in (keywords or []) if kw.strip()]
        self.tools[name] = {
            "description": description,
            "func": func,
            "keywords": normalized_keywords,
        }
        print(f"工具 '{name}' 已注册。")
        
    def getTool(self, name: str) -> callable:
        return self.tools.get(name, {}).get("func")
    
    def _tokenize(self, text: str) -> List[str]:
        return [tok.lower() for tok in re.findall(r"[\w\u4e00-\u9fff]+", text or "")]

    def selectTools(self, query: str, max_tools: int = 8) -> List[str]:
        if not query:
            return list(self.tools.keys())[:max_tools]

        query_tokens = set(self._tokenize(query))
        scored = []
        for name, info in self.tools.items():
            score = 0
            score += 3 * sum(1 for kw in info.get("keywords", []) if kw in query_tokens)
            for token in query_tokens:
                if token and token in info.get("description", "").lower():
                    score += 1
            scored.append((score, name))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [name for score, name in scored if score > 0][:max_tools]
        if selected:
            return selected
        return list(self.tools.keys())[:max_tools]

    def getAvailableTools(self, query: str = "", max_tools: int = 8) -> str:
        selected_names = self.selectTools(query=query, max_tools=max_tools)
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
            if name in selected_names
        ])
        
if __name__ == '__main__':   
    load_dotenv()
     
    toolExecutor = ToolExecutor()
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool(
        "Search",
        search_description,
        search,
        keywords=["搜索", "时事", "新闻", "事实", "网页", "internet", "search"],
    )
    calculator_description = "一个数学计算器。输入数学表达式并返回结果，支持 + - * / // % ** 和括号。"
    toolExecutor.registerTool(
        "Calculator",
        calculator_description,
        calculator,
        keywords=["数学", "计算", "表达式", "算术", "calculator", "math"],
    )
    
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    print("\n--- 执行 Action: Calculator[(123 + 456) * 789 / 12] ---")
    tool_name = "Calculator"
    tool_input = "(123 + 456) * 789 / 12"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")