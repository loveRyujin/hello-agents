from llm import HelloAgentsLLM
from tool import ToolExecutor, search, calculator
import re

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []
        self.max_tool_failures = 3
        
    def run(self, question: str):
        self.history = []
        current_step = 0
        
        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools(query=question, max_tools=8)
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            
            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break
            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 允许模型偶发漏掉结尾 "]"，避免 None.group 崩溃
                finish_match = re.match(r"Finish\[(.*)\]\s*$", action, re.DOTALL)
                if finish_match:
                    final_answer = finish_match.group(1).strip()
                elif action.startswith("Finish["):
                    final_answer = action[len("Finish["):].strip()
                else:
                    print("警告:Finish 指令格式无效，流程终止。")
                    break
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                self._record_tool_failure(
                    "动作格式错误",
                    "Action 必须为 ToolName[tool_input] 或 Finish[answer]。",
                )
                if self._too_many_tool_failures():
                    break
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
                self._record_tool_failure(
                    "工具不存在",
                    f"可用工具仅有: {', '.join(self.tool_executor.tools.keys())}。",
                )
                if self._too_many_tool_failures():
                    break
            else:
                observation = tool_function(tool_input) # 调用真实工具
                if isinstance(observation, str) and observation.startswith("错误:"):
                    self._record_tool_failure("工具执行失败", observation)
                    if self._too_many_tool_failures():
                        break
                else:
                    self._clear_recent_tool_failures()
            
            print(f"👀 观察: {observation}")
            
            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
            
        # 循环结束
        print("已达到最大步数，流程终止。")
        return None

    def _record_tool_failure(self, reason: str, detail: str):
        guidance = (
            "ToolFeedback: 你上一步工具调用失败。\n"
            f"- 原因: {reason}\n"
            f"- 细节: {detail}\n"
            "- 纠正策略:\n"
            "  1) 先检查可用工具名称是否精确匹配（区分大小写）。\n"
            "  2) Action 格式必须是 ToolName[tool_input]。\n"
            "  3) 计算问题优先使用 Calculator；事实检索问题优先使用 Search。\n"
            "  4) 若信息已足够，输出 Finish[最终答案]。"
        )
        self.history.append(guidance)

    def _too_many_tool_failures(self) -> bool:
        recent_feedback = [item for item in self.history if item.startswith("ToolFeedback:")]
        if len(recent_feedback) >= self.max_tool_failures:
            print("警告:连续多次工具调用失败，流程终止。")
            return True
        return False

    def _clear_recent_tool_failures(self):
        self.history = [item for item in self.history if not item.startswith("ToolFeedback:")]

    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None
        
if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()
        
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
        
        reactAgent = ReActAgent(llmClient, toolExecutor)
        reactAgent.run("计算 (123 + 456) × 789 / 12 = ? 的结果")
    except ValueError as e:
        print(e)
    
    