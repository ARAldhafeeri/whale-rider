import json
import argparse
from datetime import datetime
import os
import importlib.util
import subprocess
import shutil
import re
import time
import httpx 
import glob 
import string
import random
from typing import List, Dict, Any, Optional
from duckduckgo_search import DDGS
import threading

# Updated LangChain imports
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Model imports
from langchain_openai import ChatOpenAI

# Agent and tools imports
from langchain_core.tools import StructuredTool
from langchain.agents import AgentExecutor, create_openai_tools_agent

# Memory imports
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
from pydantic import BaseModel, Field 


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

load_dotenv()

# CMD styling :
BOLD = "\033[1m"
BLUE = "\033[94m"
RESET = ""
CYAN = "\033[36m"
NEW_LINE = "\n"
# Base Tool Inputs
class WebSearchInput(BaseModel):
    query: str = Field(description="Search query for web search")

class WebScrapeInput(BaseModel):
    url: str = Field(description="URL to webscrape")

class FileOpsInput(BaseModel):
    action: str = Field(description="Action to perform: read or write")
    filename: str = Field(description="Name of the file to read/write (should be in ./workspace)")
    content: Optional[str] = Field(default=None, description="Content to write (if action is 'write')")

class OsCommandInput(BaseModel):
    command: str = Field(description="Command to execute on the OS (working in ./workspace)")

class MathCalcInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")

class ExecutePythonInput(BaseModel):
    code: str = Field(description="Python code to execute")

class SystemInfoInput(BaseModel):
    dummy: str = Field(default="dummy", description="Dummy input to trigger system info retrieval")

class ReadDirectoryInput(BaseModel):
    path: str = Field(default="./workspace", description="Path to read directory contents from")

class CreateDocumentationInput(BaseModel):
    prefix: str = Field(description="Prefix for the doc type e.g. research_, tool_")
    filename: str = Field(description="Name of the file to create")
    content: str = Field(description="Content of the docs")

# Tool implementations
def web_search(query: str) -> str:
    """Perform a web search"""
    time.sleep(1)  # Simulate delay for web search
    try:
        results = DDGS().text(query, max_results=5)
        if len(results) > 0:
            return json.dumps(results, indent=2)
        return f"No results found for query: {query}"
    except Exception as e:
        return f"Web search failed: {str(e)}. Query was: {query}"

def web_scrape(url: str) -> str:
    """Scrape a webpage"""
    try:
        time.sleep(1) 
        data = httpx.get(url)
        return data.text if data.status_code == 200 else f"Failed to scrape {url}. Status code: {data.status_code}"
    except httpx.RequestError as e:
        return f"Error occurred while scraping {url}: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
    
def os_ops(command: str) -> str: 
    """Execute OS commands (restricted to ./workspace directory)"""
    try:
        # Ensure workspace directory exists
        if not os.path.exists("./workspace"):
            os.makedirs("./workspace")
            
        # Change to workspace directory before executing command
        original_dir = os.getcwd()
        os.chdir("./workspace")
        
        try:
            # Execute command in workspace directory
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = f"Command output:\n{result.stdout}\nErrors:\n{result.stderr}" if result.stdout or result.stderr else "Command executed with no output"
            return output
        finally:
            # Return to original directory
            os.chdir(original_dir)
    except Exception as e:
        return f"Command execution failed: {str(e)}"
    
def file_ops(action: str, filename: str, content: Optional[str] = None) -> str:
    """Handle file operations (restricted to ./workspace directory)"""
    try:
        # Ensure workspace directory exists
        if not os.path.exists("./workspace"):
            os.makedirs("./workspace")
            
        # Ensure filename is within workspace
        if not filename.startswith("./workspace/"):
            filename = os.path.join("./workspace", os.path.basename(filename))
            
        if action == 'read':
            if not os.path.exists(filename):
                return f"Error: File {filename} does not exist"
            with open(filename, 'r') as f:
                return f.read()
        elif action == 'write':
            if content is None:
                return "Error: 'content' is required for write operations"
            
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                f.write(content)
            return f"File {filename} written successfully"
        else:
            return "Invalid action. Use 'read' or 'write'."
    except Exception as e:
        return f"File operation error: {str(e)}"
    
def create_documentation(prefix: str, filename: str, content: str):
    """
        Tool for agent to create documentation
        prefix: prefix for the doc type e.g. research_, tool_
        filename: name of the file to create
        content: content of the docs
    """
    try:
        # Ensure workspace directory exists
        if not os.path.exists("./workspace"):
            os.makedirs("./workspace")
            
        # Ensure filename is within workspace
        if not filename.startswith("./workspace/"):
            filename = os.path.join("./workspace", os.path.basename(filename))

        # Ensure docs exists within the workspace 
        if not os.path.exists("./workspace/docs"):
            os.makedirs("./workspace/docs")
        
        # add prefix to filename 
        if prefix:
            base = os.path.basename(filename)
            new_file_name = prefix + base 

            filename = os.path.join("./workspace/docs", new_file_name)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(content)
        return f"File {filename} written successfully"
    except Exception as e:
        return f"File operation error: {str(e)}"
    

def math_calc(expression: str) -> str:
    """Evaluate a mathematical expression"""
    try:
        # Using eval with caution - in production, use a safer alternative
        # Add safety by checking if the expression contains only allowed characters
        if not re.match(r'^[\d\s\+\-\*\/\(\)\.\,\%\*\*]+$', expression):
            return "Invalid mathematical expression"
        return str(eval(expression))
    except Exception as e:
        return f"Math evaluation failed: {str(e)}"

def execute_python(code: str) -> str:
    """Execute Python code directly (in workspace environment)"""
    try:
        # Get the current directory name
        current_dir = os.path.basename(os.getcwd())
        
        # Set workspace path based on current directory
        if current_dir == "workspace":
            workspace_path = "."
        else:
            workspace_path = "./workspace"
            
        # Ensure workspace directory exists
        if not os.path.exists(workspace_path):
            os.makedirs(workspace_path)
        
        # Create a temporary script file in workspace
        script_file = f"{workspace_path}/temp_python_script.py"
        with open(script_file, 'w') as f:
            f.write(code)
        
        # Execute the script
        result = subprocess.run(['python', script_file], capture_output=True, text=True, timeout=20)
        
        # Create dir for executed python
        executed_path = f"{workspace_path}/executed_python"
        if not os.path.exists(executed_path):
            os.makedirs(executed_path)

        # Move the script to executed python dir
        shutil.copyfile(script_file, f"{executed_path}/{id_generator()}.py")
        
        # Clean up
        if os.path.exists(script_file):
            os.remove(script_file)

        if result.returncode != 0:
            return f"Error executing script: {result.stderr}"
            
        return f"Execution Result:\n{result.stdout}\nErrors:\n{result.stderr}"
    except Exception as e:
        return f"Execution failed: {str(e)}"
    

def system_info(dummy: str) -> str:
    """Get basic system information"""
    import platform
    import sys
    
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "workspace_exists": os.path.exists("./workspace"),
        "workspace_contents": os.listdir("./workspace") if os.path.exists("./workspace") else [],
        "username": os.getlogin() if hasattr(os, 'getlogin') else "unknown"
    }
    
    return json.dumps(info, indent=2)

def read_directory(path: str = "./workspace") -> str:
    """List files and directories in the specified path (default: ./workspace)"""
    try:
        # For security, restrict to workspace or its subdirectories
        if not path.startswith("./workspace"):
            path = "./workspace"
            
        # Ensure workspace exists
        if not os.path.exists("./workspace"):
            os.makedirs("./workspace")
            
        if not os.path.exists(path):
            return f"Path {path} does not exist"
            
        items = os.listdir(path)
        result = []
        
        for item in items:
            full_path = os.path.join(path, item)
            item_type = "file" if os.path.isfile(full_path) else "directory"
            size = os.path.getsize(full_path) if os.path.isfile(full_path) else "-"
            
            result.append({
                "name": item,
                "type": item_type,
                "size": size
            })
            
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error reading directory: {str(e)}"

class AutonomousAgent:
    def __init__(self, objective, model, api_key, api_base, max_iterations=5, verbose=False, auto=False):
        self.memory_file = "./workspace/agent_memory.json"
        self.objective = objective
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.goals = []
        self.verbose = verbose
        self.human_feedback = ""
        self.auto = auto
        
        # Ensure workspace directory exists
        if not os.path.exists("./workspace"):
            os.makedirs("./workspace")
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(api_key=api_key, base_url=api_base, model=model, temperature=0.80,   max_tokens=15000)
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
        # Create vector store for persistent memory
        self.vector_store = chromadb.PersistentClient(path="./workspace/chroma_db")
        self.core_collection = self.vector_store.get_or_create_collection("workspace")
        
        # Base tools setup
        self.base_tools = self._load_base_tools()
        self.custom_tools = {}
        

        self.execution_locks = {}  # Thread locks for each tool
        self.worker_threads = {}   # Worker threads pool
        self.max_threads = os.cpu_count() or 4  # Default to machine's CPU count
        self.thread_semaphore = threading.Semaphore(self.max_threads)  # Control concurrent threads
        self.results_lock = threading.Lock()  # Lock for writing to shared results
        # Initialize agent memory
        self._initialize_memory()

        # Add new sections to memory structure
        if 'tool_execution_plans' not in self.agent_memory:
            self.agent_memory['tool_execution_plans'] = []
        
        if 'execution_history' not in self.agent_memory:
            self.agent_memory['execution_history'] = []

    def _initialize_memory(self):
        """Initialize or load agent memory"""
        if not os.path.exists(self.memory_file):
            self.agent_memory = {
                "system": {
                    "objective": self.objective,
                    "status": "IN_PROGRESS",
                    "created_at": datetime.now().isoformat(),
                    "iterations": []
                },
                "goals": [],
                "tools": [],
                "reflections": [],
                "human_feedback": [],
                "task_analysis": []
            }
        else:
            with open(self.memory_file, 'r') as f:
                self.agent_memory = json.load(f)
                # Clear previous state to start fresh if objective changed
                if self.agent_memory["system"]["objective"] != self.objective:
                    self.agent_memory = {
                        "system": {
                            "objective": self.objective,
                            "status": "IN_PROGRESS",
                            "created_at": datetime.now().isoformat(),
                            "iterations": []
                        },
                        "goals": [],
                        "tools": [],
                        "reflections": [],
                        "human_feedback": []
                    }

    def _load_base_tools(self) -> List[StructuredTool]:
        """Load base tools as LangChain Tool objects"""
        tools = [
            StructuredTool.from_function(
                name="web_search",
                func=web_search,
                args_schema=WebSearchInput,
                description="Search the web for information. Input should be a search query."
            ),
            StructuredTool.from_function(
                name="file_operations",
                func=file_ops,
                args_schema=FileOpsInput,
                description="Read or write files in the ./workspace directory. Input should include: action (read/write), filename, and content (for write operations)."
            ),
            StructuredTool.from_function(
                name="run_os_command",
                func=os_ops,
                args_schema=OsCommandInput,
                description="Execute OS commands within the ./workspace directory. Input should be a command string."
            ),
            StructuredTool.from_function(
                name="math_calculation",
                func=math_calc,
                args_schema=MathCalcInput,
                description="Evaluate mathematical expressions. Input should be a valid mathematical expression as a string."
            ),
            StructuredTool.from_function(
                name="execute_python",
                func=execute_python,
                args_schema=ExecutePythonInput,
                description="Execute a Python script within the ./workspace directory. Input should be Python code as a string."
            ),
            StructuredTool.from_function(
                name="system_info",
                func=system_info,
                args_schema=SystemInfoInput,
                description="Get basic system information about the environment."
            ),
            StructuredTool.from_function(
                name="read_directory",
                func=read_directory,
                args_schema=ReadDirectoryInput,
                description="List files and directories in the specified path (default is ./workspace)."
            ),
            StructuredTool.from_function(
                name="web_scrape",
                func=web_scrape,
                args_schema=WebScrapeInput,
                description="Scrape a webpage. Input should be a valid URL. This is only GET Request."
            ),
            StructuredTool.from_function(
                name="create_documentation",
                func=create_documentation,
                args_schema=CreateDocumentationInput,
                description="Create documentation files in the ./workspace/docs directory. Input should include: prefix, filename, and content."
            ),
        ]
        return tools

    def save_memory(self):
        """Save agent memory to file and update vector store with new information"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        
        with open(self.memory_file, 'w') as f:
            json.dump(self.agent_memory, f, indent=2)
        
        # Update vector store with new memory entries
        memory_text = json.dumps(self.agent_memory, indent=2)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(memory_text)
        ids = [f"memory_chunk_{id_generator(10)}" for _ in range(len(texts))]
        metadata = [
            {"source": f"memory_chunk_{id_generator(10)}", "timestamp": datetime.now().isoformat()} for i in range(len(texts))]
        # Store memory chunks with metadata
        embeddings = self.embedding_function(texts)
        self.core_collection.add(ids, embeddings, metadatas=metadata)


    def generate_initial_goals(self):
        """Generate a focused set of initial goals based on the objective"""
        prompt = PromptTemplate.from_template(
            """
            Break down this objective into sequential, focused goals: {objective}
            Make sure the goals acheive the objective
            Remember that it's better to have fewer, well-defined goals than too many broad ones.
            Make sure the goals are well defined with specific outcomes and clear steps to achieve them.
            Each goal should be concrete, measurable, and directly contributes to the main objective.
            Add goals to check if goals are completed or not based on the objective.
            The goals you generate should ressible the objective to acheive.
            Return a JSON array of goal objects with 'description' and 'priority' fields.
            Priority should be a number indicating the importance of the goal (1 being the highest priority).
            IMPORTANT :
                for every completed goal, please call the documentation tool and output documnetation
                The output should be detailed and include detailed summary of whatever the output of 
                acheived goal is.
            """
        )
        
        # Create a runnable chain with the prompt and LLM
        goal_chain = prompt | self.llm | StrOutputParser()
        
        try:
            # Invoke the chain
            response = goal_chain.invoke({"objective": self.objective})
            
            # Parse the goals from the response
            json_match = re.search(r'\[.*\]', response.replace('\n', ' '), re.DOTALL)
            if json_match:
                response = json_match.group()
            
            new_goals = json.loads(response)
            
            self._add_goal("Break down this objective into sequential, focused goals that represent full plan to acehieve the objective", 1)
            priority_map = {
                "high": 1,
                "medium": 2,
                "low": 3,
            }            
            for i, goal in enumerate(new_goals):
                priority = goal.get('priority', 1)
                if isinstance(priority, str):
                    priority = priority_map.get(priority.lower(), 1)
                self._add_goal(goal['description'], )
            
            self.log(f"Generated {len(new_goals)} focused initial goals")
        except Exception as e:
            self.log_error(f"Failed to parse goals: {str(e)}")
            # Add a single default goal if parsing fails

    def create_tool(self, description: str, input_schema=None) -> bool:
        """Create a custom tool using LangChain"""
        prompt = PromptTemplate(
            template="""
            Create a Python function to help achieve this: {description}
            IMPORTANT : 
                - Only create the tool if there is no existing tool that can do this.
            Requirements:
            - Function should work within the ./workspace directory
            - Use only standard libraries 
            - Include error handling
            - Return meaningful status messages
            - Function name should be descriptive: {tool_name}
            
            For the input schema:
            ```python
            from pydantic import BaseModel, Field 
            class {tool_name}Input(BaseModel):
                param1: str = Field(description="Description of param1")
                # Add other parameters as needed
            ```
            
            The function should respect the workspace directory restriction.
            """
        )
        
        # Generate a tool name based on description
        tool_name = f"tool_{len(self.custom_tools) + 1}"
        
        # Create a runnable chain
        tool_chain = prompt | self.llm | StrOutputParser()
        
        try:
            # Invoke the chain
            response = tool_chain.invoke({
                "description": description, 
                "tool_name": tool_name
            })
            
            code = self._extract_code(response)
            
            # Create a module for the dynamic tool
            spec = importlib.util.spec_from_loader('dynamic_tool', loader=None)
            module = importlib.util.module_from_spec(spec)
            
            # Execute the code in the module's namespace
            exec(code, module.__dict__)
            
            # Find the function and input schema in the module
            function_name = None
            schema_name = None
            
            for name in dir(module):
                if callable(getattr(module, name)) and not name.startswith('__'):
                    function_name = name
                elif name.endswith('Input') and not name.startswith('__'):
                    schema_name = name
            
            if not function_name:
                self.log_error(f"Tool function not found in generated code")
                return False
                
            tool_func = getattr(module, function_name)
            tool_schema = getattr(module, schema_name) if schema_name else None
            
            # Create a LangChain Tool from the custom function
            if tool_schema:
                custom_tool = StructuredTool.from_function(
                    name=tool_name,
                    func=tool_func,
                    args_schema=tool_schema,
                    description=f"Custom tool for: {description}"
                )
            else:
                custom_tool = StructuredTool.from_function(
                    name=tool_name,
                    func=tool_func,
                    description=f"Custom tool for: {description}"
                )
            
            self.custom_tools[tool_name] = custom_tool
            
            self.agent_memory['tools'].append({
                "name": tool_name,
                "code": code,
                "created_at": datetime.now().isoformat(),
                "purpose": description
            })
            
            self.log(f"Created new tool: {tool_name} for {description}")
            return True
        except Exception as e:
            self.log_error(f"Tool creation failed: {str(e)}")
            return False
        
    def _update_goal_status(self, goal_id: int, status: str, result: str = None):
        """Enhanced goal status update with progress tracking"""
        for idx, goal in enumerate(self.goals):
            if goal["id"] == goal_id:
                self.goals[idx]["status"] = status
                self.goals[idx]["completed_at"] = datetime.now().isoformat() if status in ["COMPLETED", "FAILED"] else None
                self.goals[idx]["result"] = result if result else ""
                
                # Update in agent_memory
                for mem_idx, mem_goal in enumerate(self.agent_memory['goals']):
                    if mem_goal["id"] == goal_id:
                        self.agent_memory['goals'][mem_idx]["status"] = status
                        self.agent_memory['goals'][mem_idx]["completed_at"] = self.goals[idx]["completed_at"]
                        self.agent_memory['goals'][mem_idx]["result"] = self.goals[idx]["result"]
                        break
                        
                # Trigger goal dependencies
                self._check_dependent_goals(goal_id)
                break

    def _add_goal(self, description: str, priority: int = 1, parent: Optional[int] = None, depends_on: List[int] = None):
        """Add a new goal with dependencies"""
        goal_id = len(self.agent_memory['goals']) + 1
        goal = {
            "id": goal_id,
            "description": description,
            "status": "PENDING",
            "priority": priority,
            "parent": parent,
            "depends_on": depends_on if depends_on else [],
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "result": None
        }
        self.agent_memory['goals'].append(goal)
        self.goals.append(goal)
        self.log(f"Added new goal: {description} (priority: {priority})")
        return goal_id

    def _check_dependent_goals(self, completed_goal_id):
        """Check and update status of goals dependent on the completed goal"""
        for goal in self.goals:
            if goal["status"] == "PENDING" and goal.get("depends_on") and completed_goal_id in goal.get("depends_on", []):
                # Check if all dependencies are satisfied
                all_deps_met = True
                for dep_id in goal.get("depends_on", []):
                    dep_goal = next((g for g in self.goals if g["id"] == dep_id), None)
                    if not dep_goal or dep_goal["status"] != "COMPLETED":
                        all_deps_met = False
                        break
                
                if all_deps_met:
                    self.log(f"Goal {goal['id']} dependencies satisfied, now ready: {goal['description']}", "INFO")
                    # Mark ready by increasing priority
                    goal["priority"] = 1  # Highest priority

    def process_goals(self):
        """Process pending goals with better selection criteria"""
        # Filter goals that are ready to be processed (no pending dependencies)
        ready_goals = []
        for goal in self.goals:
            if goal['status'] == 'PENDING':
                dependencies_met = True
                for dep_id in goal.get("depends_on", []):
                    dep_goal = next((g for g in self.goals if g["id"] == dep_id), None)
                    if not dep_goal or dep_goal["status"] != "COMPLETED":
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    ready_goals.append(goal)
        
        if not ready_goals:
            self.log("No ready goals to process", "WARNING")
            return
        
        # Sort by priority then by creation time (older first)
        ready_goals.sort(key=lambda x: (x['priority'], datetime.fromisoformat(x['created_at']).timestamp()))
        
        # Process the highest priority goal
        goal = ready_goals[0]
        self.log(f"Processing goal: {goal['description']} (priority: {goal['priority']})")
        
        result = self._execute_goal(goal)
        
        if result:
            self._update_goal_status(goal['id'], 'COMPLETED')
            self.log(f"Goal completed: {goal['description']}", "INFO")
        else:
            self._update_goal_status(goal['id'], 'FAILED')
            self.log(f"Goal failed: {goal['description']}", "WARNING")
        
        # Reflect on progress after completing/failing the goal
        self.reflect_on_progress(goal, result)

    def recommend_tools(self, goal_description: str) -> List[str]:
        """Recommend appropriate tools for a specific goal with improved selection"""
        prompt = PromptTemplate(
            input_variables=["goal_description", "tools"],
            template="""
    For this goal: {goal_description}

    These tools are available:
    {tools}

    IMPORTANT: Analyze the goal carefully and provide a COMPREHENSIVE set of tools needed to complete it.
    Don't just pick the most obvious tool - think about the complete workflow to achieve the goal.

    Consider:
    1. What data needs to be gathered? (web_search, web_scrape, read_directory, file_operations)
    2. What processing is needed? (execute_python, math_calculation)
    3. What system operations might help? (run_os_command, system_info)
    4. How will results be stored? (file_operations, create_documentation)

    Return a JSON object with:
    1. "tools": Array of recommended tool names
    2. "reasoning": Brief justification for each tool
    3. "execution_order": Suggested order of tool execution for maximum efficiency

    Example format:
    {{
    "tools": ["web_search", "web_scrape", "execute_python", "file_operations"],
    "reasoning": "Need to search for data, scrape details, process with Python, and save results",
    "execution_order": ["web_search", "web_scrape", "execute_python", "file_operations"]
    }}
    """
        )
        
        tool_descriptions = []
        for tool in self.base_tools + list(self.custom_tools.values()):
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        
        # Create a runnable chain
        recommend_chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = recommend_chain.invoke({
                "goal_description": str(goal_description),
                "tools": str(tool_descriptions),
            })
            
            # Extract tool names from response
            try:
                # Parse JSON response
                json_match = re.search(r'\{.*\}', response.replace('\n', ''), re.DOTALL)
                if json_match:
                    tool_data = json.loads(json_match.group())
                    recommended_tools = tool_data.get("tools", [])
                    # Store execution order and reasoning in agent memory for later use
                    self.agent_memory['tool_execution_plans'].append({
                        "goal": goal_description,
                        "recommended_tools": recommended_tools,
                        "reasoning": tool_data.get("reasoning", ""),
                        "execution_order": tool_data.get("execution_order", recommended_tools),
                        "timestamp": datetime.now().isoformat()
                    })
                    self.log(f"Tool recommendation: {', '.join(recommended_tools)}", "INFO")
                    return recommended_tools
                else:
                    # Simple extraction fallback
                    tool_names = []
                    for line in response.split('\n'):
                        if line.strip().startswith('-') or line.strip().startswith('*'):
                            tool_name = line.strip().split(':')[0].strip('-* "\'')
                            tool_names.append(tool_name)
                    return tool_names
            except:
                self.log(f"Could not parse tool recommendations: {response}", "WARNING")
                return self._extract_tool_names_from_text(response)
        except Exception as e:
            self.log(f"Tool recommendation failed: {str(e)}", "ERROR")
            return []
            
    def _extract_tool_names_from_text(self, text):
        """Extract tool names from text when JSON parsing fails"""
        tool_names = []
        all_tool_names = [tool.name for tool in self.base_tools] + list(self.custom_tools.keys())
        
        for tool_name in all_tool_names:
            if tool_name in text:
                tool_names.append(tool_name)
        
        return tool_names[:5] 
        

    def save_checkpoint(self):
        """Save a complete checkpoint of agent state for recovery"""
        try:
            checkpoint_dir = os.path.join("./workspace", "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint = {
                "objective": self.objective,
                "current_iteration": self.current_iteration,
                "max_iterations": self.max_iterations,
                "goals": self.goals,
                "agent_memory": self.agent_memory,
                "human_feedback": self.human_feedback,
                "custom_tools": {name: {
                    "name": tool.name,
                    "description": tool.description,
                    "code": next((t["code"] for t in self.agent_memory["tools"] if t["name"] == name), None)
                } for name, tool in self.custom_tools.items()},
                "timestamp": datetime.now().isoformat()
            }
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
                
            # Keep only last 5 checkpoints
            checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.json")))
            if len(checkpoints) > 5:
                for old_checkpoint in checkpoints[:-5]:
                    os.remove(old_checkpoint)
                    
            self.log(f"Checkpoint saved to {checkpoint_path}", "INFO")
            return checkpoint_path
        except Exception as e:
            self.log(f"Checkpoint save failed: {str(e)}", "ERROR")
            return None


    def _execute_goal(self, goal):
        """Execute a goal using multiple tools in sequence based on recommendations with improved threading"""
        try:
            # Extract goal information
            goal_description = goal.get("description", "No description provided")
            goal_id = goal.get("id", 0)
            
            # Get recommended tools for this goal
            recommended_tools = self.recommend_tools(goal_description)
            
            # Color mapping for tool outputs
            colors = {
                "web_search": "\033[1;94m",      # Bold Blue
                "web_scrape": "\033[1;96m",      # Bold Cyan
                "file_operations": "\033[1;92m",  # Bold Green
                "run_os_command": "\033[1;91m",   # Bold Red
                "math_calculation": "\033[1;95m", # Bold Magenta
                "execute_python": "\033[1;93m",   # Bold Yellow
                "system_info": "\033[1;97m",      # Bold White
                "read_directory": "\033[1;90m",   # Bold Gray
                "create_documentation": "\033[1;92m", # Bold Green
                "code_generator": "\033[1;96m",   # Bold Cyan
                "code_analyzer": "\033[1;95m",    # Bold Magenta
            }
            default_color = ""  # Reset color
            
            # Initialize rich display elements
            print(f"\n{'='*80}")
            print(f"🚀 \033[1;97mEXECUTING GOAL: {goal_description}")
            print(f"{'='*80}\n")
            
            # Execute each recommended tool in sequence
            results = {}
            execution_summary = []
            all_raw_outputs = {}  # Store complete outputs
            
            self.log(f"Executing goal with {len(recommended_tools)} tools: {', '.join(recommended_tools)}", "INFO")
            
            # Get existing execution plan if available
            execution_plans = [plan for plan in self.agent_memory.get('tool_execution_plans', []) 
                            if plan.get('goal') == goal_description]
            
            is_coding_task = self._is_coding_task(goal_description)
            # Use the execution order from plan if available
            if execution_plans and 'execution_order' in execution_plans[-1]:
                tool_execution_order = execution_plans[-1]['execution_order']
            else:
                # Determine if this is a coding-focused task
                
                if is_coding_task:
                    # For coding tasks, prioritize code generation and analysis tools
                    tool_execution_order = self._optimize_tool_order_for_coding(recommended_tools)
                else:
                    # Use recommended order for research tasks
                    tool_execution_order = recommended_tools
                
            # Initialize progress tracking
            total_tools = len(tool_execution_order)
            completed_tools = 0
            
            # For coding tasks, add initial planning step
            if is_coding_task:
                print(f"\n{BOLD}🔍 ANALYZING CODING TASK")
                print(f"{'▼'*50}")
                coding_plan = self._generate_coding_plan(goal_description)
                print(f"\033[1;96m📋 CODING PLAN:\n{coding_plan}\033[0m")
                print(f"{'▲'*50}\n")
            
            # Execute tools in the determined order
            for tool_name in tool_execution_order:
                # Find the tool
                tool = None
                for base_tool in self.base_tools:
                    if base_tool.name == tool_name:
                        tool = base_tool
                        break
                
                if not tool and tool_name in self.custom_tools:
                    tool = self.custom_tools[tool_name]
                    
                if not tool:
                    self.log(f"Tool '{tool_name}' not found, skipping", "WARNING")
                    continue
                    
                # Display progress with enhanced visuals
                completed_tools += 1
                progress_pct = (completed_tools / total_tools) * 100
                progress_bar = "█" * int(progress_pct / 5) + "░" * (20 - int(progress_pct / 5))
                
                print(f"\n{BOLD}[{completed_tools}/{total_tools}] ({progress_pct:.1f}%)")
                print(f"📊 \033[1;36mProgress: |{progress_bar}|")
                print(f"🔧 {BOLD}Running tool: \033[1;93m{tool_name}")
                
                # Generate input for this tool based on previous results
                tool_input = self._generate_tool_input(tool, goal_description, results, is_coding_task)
                
                # Execute the tool with enhanced visual display
                color = colors.get(tool_name, default_color)
                print(f"\n{color}🔄 EXECUTING: {tool_name}{default_color}")
                print(f"{color}{'▼'*50}{default_color}")
                
                # Execute and capture result
                start_time = time.time()
                try:
                    # Acquire lock for this tool to prevent race conditions
                    if tool_name not in self.execution_locks:
                        self.execution_locks[tool_name] = threading.Lock()
                        
                    with self.execution_locks[tool_name]:
                        result = self._execute_single_tool(tool, tool_input)
                    
                    execution_time = time.time() - start_time
                    
                    # Store result
                    with self.results_lock:
                        results[tool_name] = result
                        all_raw_outputs[tool_name] = result
                    
                    # Process code results for better coherence
                    if tool_name in ["execute_python", "code_generator"] and isinstance(result, str):
                        # Extract clean code from output
                        result = self._process_code_output(result, tool_name)
                        
                    # Add to summary
                    execution_summary.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "success": True,
                        "execution_time": execution_time,
                        "result_snippet": result[:300] if isinstance(result, str) else str(result)[:300]
                    })
                    
                    # Display colored preview with enhanced formatting
                    result_preview = result[:300] + "..." if isinstance(result, str) and len(result) > 300 else result
                    print(f"{color}✓ SUCCESS [{execution_time:.2f}s]{default_color}")
                    print(f"{color}📋 RESULT PREVIEW:{default_color}")
                    print(f"{color}{'-'*50}{default_color}")
                    print(f"{color}{result_preview}{default_color}")
                    print(f"{color}{'-'*50}{default_color}")
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    error_msg = f"Error executing {tool_name}: {str(e)}"
                    self.log(error_msg, "ERROR")
                    
                    # Add to summary
                    execution_summary.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "success": False,
                        "execution_time": execution_time,
                        "error": str(e)
                    })
                    
                    # Display error
                    print(f"\033[91m❌ FAILED [{execution_time:.2f}s]: {str(e)}")
            
            # If this was a coding task, generate a final code summary
            if is_coding_task:
                outputs = self._generate_coding_focused_outputs(goal, recommended_tools, execution_summary, results, all_raw_outputs)
            else:
                # For research tasks, use the original comprehensive outputs
                outputs = self._generate_comprehensive_outputs(goal, recommended_tools, execution_summary, results, all_raw_outputs)
            
            # Create documentation
            self._create_execution_documentation(goal, recommended_tools, execution_summary, results, outputs)
            
            # Store results in memory
            self._store_goal_results(goal, recommended_tools, results, execution_summary, outputs)
            
            # Display completion message
            print(f"\n{'='*80}")
            print(f"✅ \033[1;92mGOAL EXECUTION COMPLETE")
            print(f"📁 {BOLD}Outputs generated: {', '.join(outputs.keys())}")
            print(f"{'='*80}\n")
            
            return outputs
        except Exception as e:
            self.log_error(f"Goal execution failed: {str(e)}")
            return False

    def _is_coding_task(self, goal_description: str) -> bool:
        """
        Determine if a task is coding-focused using LLM analysis with structured response.
        
        Args:
            goal_description: The description of the task/goal to analyze
            
        Returns:
            bool: True if this is a coding task, False otherwise
        """
        prompt = PromptTemplate(
            input_variables=["goal_description"],
            template="""
    Analyze the following task description and determine if it requires significant coding work:

    Task: {goal_description}

    Consider these indicators of coding tasks:
    1. Requires writing or modifying code (scripts, functions, classes)
    2. Involves programming concepts (algorithms, data structures)
    3. Needs software development (applications, libraries, APIs)
    4. Requires debugging or optimizing existing code
    5. Involves technical implementation details

    Return a JSON response with:
    1. "is_coding_task": boolean (true/false)
    2. "confidence": float (0.0-1.0) indicating confidence level
    3. "indicators": array of strings explaining what suggests coding is needed
    4. "required_skills": array of relevant programming skills if applicable

    Example responses:
    {{
        "is_coding_task": true,
        "confidence": 0.9,
        "indicators": ["mentions 'write a Python script'", "requires API development"],
        "required_skills": ["Python", "FastAPI"]
    }}

    {{
        "is_coding_task": false,
        "confidence": 0.95,
        "indicators": ["only requires data analysis without implementation"],
        "required_skills": []
    }}
    """
        )
        
        # Create analysis chain
        analysis_chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = analysis_chain.invoke({
                "goal_description": str(goal_description)
            })
            
            # Parse JSON response
            try:
                json_match = re.search(r'\{.*\}', response.replace('\n', ''), re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    
                    # Store analysis in agent memory
                    try:
                        self.agent_memory['task_analysis'].append({
                        "goal": goal_description,
                        "analysis": analysis,
                        "timestamp": datetime.now().isoformat()
                    })
                    except:
                        print("FAILED to update memory with coding anlaysis")
                    
                    # Log the decision
                    self.log(
                        f"Coding task analysis: {analysis.get('is_coding_task', False)} "
                        f"(confidence: {analysis.get('confidence', 0)})",
                        "INFO"
                    )
                    
                    return bool(analysis.get('is_coding_task', False))
                
                # Fallback to keyword matching if JSON parsing fails
                return self._basic_coding_task_check(goal_description)
                
            except json.JSONDecodeError:
                self.log(f"Could not parse coding task analysis: {response}", "WARNING")
                return self._basic_coding_task_check(goal_description)
                
        except Exception as e:
            self.log(f"Coding task analysis failed: {str(e)}", "ERROR")
            return self._basic_coding_task_check(goal_description)


    def _basic_coding_task_check(self, description: str) -> bool:
        """Fallback keyword-based coding task detection"""
        coding_phrases = {
            "write code", "create a program", "develop software",
            "build an app", "implement an algorithm", "write a script",
            "create a library", "develop a module", "debug the code",
            "refactor the", "optimize the", "programming solution"
        }
        description_lower = description.lower()
        return any(phrase in description_lower for phrase in coding_phrases)
   
   
    def _optimize_tool_order_for_coding(self, recommended_tools):
        """Optimize tool execution order for coding tasks"""
        # Define preferred order for coding tasks
        coding_order_priority = [
            "code_analyzer",           # First analyze requirements
            "web_search",              # Search for relevant information
            "read_directory",          # Understand existing code structure
            "file_operations",         # Read existing files
            "code_generator",          # Generate code
            "execute_python",          # Test the code
            "run_os_command",          # Run system commands if needed
            "create_documentation",    # Document the result
            "web_scrape",              # Lower priority for coding tasks
            "math_calculation",
            "system_info"
        ]
        
        # Sort tools based on priority for coding tasks
        return sorted(
            recommended_tools,
            key=lambda x: coding_order_priority.index(x) if x in coding_order_priority else 999
        )
    
    def _generate_coding_plan(self, goal_description):
        """Generate a detailed coding plan for the goal"""
        # Define a prompt template for coding plan
        prompt_template = """
        I need to create a detailed coding plan for implementing the following:
        
        GOAL: {goal_description}
        
        Please provide a detailed plan including:
        1. Problem analysis and understanding
        2. Proposed architecture or design 
        3. Key components/modules needed
        4. Implementation steps in sequence
        5. Testing strategy
        6. Potential challenges and solutions
        
        Format this as a clear, structured plan that would guide efficient code implementation.
        """
        
        # Execute the plan generation with LLM
        try:
            from langchain.prompts import PromptTemplate
            from langchain.schema.output_parser import StrOutputParser
            
            prompt = PromptTemplate(template=prompt_template, input_variables=["goal_description"])
            plan_chain = prompt | self.llm | StrOutputParser()
            
            # Generate the coding plan
            coding_plan = plan_chain.invoke({"goal_description": goal_description})
            return coding_plan
        except Exception as e:
            self.log(f"Error generating coding plan: {str(e)}", "WARNING")
            return "Could not generate coding plan. Proceeding with execution."

    def _process_code_output(self, raw_output, tool_name):
        """Process and clean code output for better coherence"""
        # Extract code blocks from the output
        code_block_pattern = r'```(?:\w+)?\s*([\s\S]*?)\s*```'
        code_blocks = re.findall(code_block_pattern, raw_output)
        
        if code_blocks:
            # Join all code blocks with clear separation
            processed_code = "\n\n".join(code_blocks)
            return processed_code
        
        # If no code blocks found but contains Python-like patterns, try to extract
        if re.search(r'def\s+\w+\s*\(', raw_output) or re.search(r'class\s+\w+\s*\:', raw_output):
            # This looks like code but wasn't in code blocks
            lines = raw_output.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if re.match(r'^\s*(def|class|import|from|if|for|while|return|#)\s', line):
                    in_code = True
                    code_lines.append(line)
                elif in_code and line.strip() and not line.startswith('#'):
                    code_lines.append(line)
                elif in_code and not line.strip():
                    code_lines.append('')  # Keep empty lines in code
                elif in_code:
                    in_code = False
            
            if code_lines:
                return '\n'.join(code_lines)
        
        # Return original if no processing applied
        return raw_output

    def _generate_tool_input(self, tool, goal_description, previous_results, is_coding_task):
        """Intelligently generate input for a tool based on previous results"""
        # Extract tool name and schema
        tool_name = tool.name
        tool_schema = getattr(tool, 'args_schema', None)
    
        # For coding tasks, use specialized input generation
        if is_coding_task and tool_name in ["code_generator", "execute_python"]:
            return self._generate_coding_tool_input(tool_name, goal_description, previous_results)
        
        # If we have no schema, return default values
        if not tool_schema:
            if tool_name == "system_info":
                return "dummy"
            return goal_description
            
        # Get expected parameters from schema
        expected_params = {}
        for field_name, field in tool_schema.model_fields.items():
            expected_params[field_name] = {
                "title": field.title,
                "description": field.description,
                "metadata": field.metadata,
            }
        
        # Import required components
        from langchain.prompts import PromptTemplate
        from langchain.schema.output_parser import StrOutputParser
        
        prompt = PromptTemplate(template="""
        I need to generate input parameters for a tool called: {tool_name}

        The tool's purpose is: {tool_description}

        The goal I'm trying to achieve is: {goal}

        The tool requires these parameters:
        {expected_params}

        Previous tool results that might be relevant:
        {previous_results}

        Please generate the appropriate input values for this tool as a JSON object.
        Only include the parameter values, not explanations.

        Example format:
        {{
            "param1": "value1",
            "param2": "value2"
        }}
        """, input_variables=["tool_name", "tool_description", "goal", "expected_params", "previous_results"])
        
        # Format previous results for the prompt
        prev_results_formatted = ""
        for prev_tool, result in previous_results.items():
            prev_results_formatted += f"\n--- {prev_tool} RESULT ---\n"
            prev_results_formatted += f"{result[:500]}..." if isinstance(result, str) and len(result) > 500 else str(result)
        
        # Execute the prompt
        input_chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = input_chain.invoke({
                "tool_name": tool_name,
                "tool_description": tool.description,
                "goal": goal_description,
                "expected_params": json.dumps(expected_params, indent=2),
                "previous_results": prev_results_formatted if prev_results_formatted else "No previous results available."
            })
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.replace('\n', ''), re.DOTALL)
            if json_match:
                params = json.loads(json_match.group())
                return params
            else:
                # Create basic parameters based on tool type as fallback
                if tool_name == "web_search":
                    return {"query": goal_description}
                elif tool_name == "web_scrape":
                    # Try to extract URL from previous search results
                    for prev_tool, result in previous_results.items():
                        if prev_tool == "web_search" and "http" in result:
                            urls = re.findall(r'https?://[^\s"\']+', result)
                            if urls:
                                return {"url": urls[0]}
                    return {"url": "https://example.com"}
                elif tool_name == "file_operations":
                    return {"action": "read", "filename": "./workspace/data.txt"}
                elif tool_name == "create_documentation":
                    return {
                        "prefix": "task_",
                        "filename": f"report_{datetime.now().strftime('%Y%m%d')}.md",
                        "content": f"# Goal Execution Report\n\n## Goal\n{goal_description}\n\n## Results\n\nExecution completed at {datetime.now().isoformat()}"
                    }
                else:
                    # Generic fallback
                    return {"dummy": "placeholder"}
        except Exception as e:
            self.log(f"Error generating tool input: {str(e)}", "ERROR")
            # Return basic input as fallback
            if tool_name == "system_info":
                return {"dummy": "dummy"}
            return {"generic_input": goal_description}

    def _generate_coding_tool_input(self, tool_name, goal_description, previous_results):
        """Generate specialized input for coding tools"""
        from langchain.prompts import PromptTemplate
        from langchain.schema.output_parser import StrOutputParser
        
        # For code generator tool
        if tool_name == "code_generator":
            prompt = PromptTemplate(template="""
            I need to generate code to solve this problem:
            
            GOAL: {goal}
            
            Based on previous tool results:
            {previous_results}
            
            Please provide detailed specifications for generating code:
            1. The programming language to use
            2. What the code should accomplish (be specific)
            3. Any specific libraries or frameworks to use
            4. Input/output specifications
            5. Any specific implementation details
            
            Format as JSON with these exact keys:
            {{
                "language": "python",
                "task": "detailed task description",
                "libraries": ["lib1", "lib2"],
                "specs": "detailed implementation specs"
            }}
            """, input_variables=["goal", "previous_results"])
        
        # For code execution tool
        elif tool_name == "execute_python":
            prompt = PromptTemplate(template="""
            I need to execute Python code to solve this goal:
            
            GOAL: {goal}
            
            Based on previous tool results, especially any code that was generated:
            {previous_results}
            
            Please extract or create the Python code to execute, and provide any necessary inputs.
            Format as JSON with these keys:
            {{
                "code": "full python code to execute",
                "inputs": {{"input1": "value1"}},
                "timeout": 30
            }}
            """, input_variables=["goal", "previous_results"])
        
        # Format previous results for the prompt
        prev_results_formatted = ""
        code_found = False
        
        # First check for any code in previous results
        for prev_tool, result in previous_results.items():
            if prev_tool in ["code_generator", "execute_python"]:
                prev_results_formatted = f"\n--- CODE FROM {prev_tool} ---\n{result}\n"
                code_found = True
                break
        
        # If no code found, include all relevant tool results
        if not code_found:
            for prev_tool, result in previous_results.items():
                prev_results_formatted += f"\n--- {prev_tool} RESULT ---\n"
                prev_results_formatted += f"{result[:500]}..." if isinstance(result, str) and len(result) > 500 else str(result)
        
        # Execute the prompt
        input_chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = input_chain.invoke({
                "goal": goal_description,
                "previous_results": prev_results_formatted if prev_results_formatted else "No previous results available."
            })
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.replace('\n', ''), re.DOTALL)
            if json_match:
                try:
                    params = json.loads(json_match.group())
                    return params
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract code blocks directly
                    if tool_name == "execute_python":
                        code_blocks = re.findall(r'```(?:python)?\s*([\s\S]*?)\s*```', response)
                        if code_blocks:
                            return {"code": code_blocks[0], "timeout": 30}
            
            # Fallbacks for specific tools
            if tool_name == "code_generator":
                return {"language": "python", "task": goal_description, "libraries": [], "specs": ""}
            elif tool_name == "execute_python":
                # Try to find Python code in any previous result
                for _, result in previous_results.items():
                    if isinstance(result, str):
                        code_blocks = re.findall(r'```(?:python)?\s*([\s\S]*?)\s*```', result)
                        if code_blocks:
                            return {"code": code_blocks[0], "timeout": 30}
                return {"code": "print('No code to execute')", "timeout": 10}
            
        except Exception as e:
            self.log(f"Error generating coding tool input: {str(e)}", "ERROR")
            # Return basic fallbacks
            if tool_name == "code_generator":
                return {"language": "python", "task": goal_description, "libraries": [], "specs": ""}
            elif tool_name == "execute_python":
                return {"code": "print('Error preparing code execution')", "timeout": 10}

    def _execute_single_tool(self, tool, tool_input):
        """Execute a single tool with proper error handling and timing"""
        # Use threading to avoid blocking
        with self.thread_semaphore:
            try:
                # Convert tool_input to appropriate format
                if isinstance(tool_input, dict):
                    # For structured tools
                    result = tool.func(**tool_input)
                else:
                    # For simple tools
                    result = tool.func(tool_input)
                
                return result
            except Exception as e:
                raise Exception(f"Tool execution error: {str(e)}")

    def _generate_coding_focused_outputs(self, goal, tools_used, execution_summary, results, raw_outputs):
        """Generate comprehensive code outputs with enhanced generation capabilities"""
        try:
            outputs = {}
            goal_description = goal.get("description", "No description provided")
            
            # Extract and format code fragments from raw outputs
            code_results_formatted = self._format_code_fragments(raw_outputs)
            
            # 1. Code Expansion Phase
            expanded_code = self._expand_code_fragments(code_results_formatted, goal_description)
            
            # 2. Generate Complete Solution
            complete_solution = self._generate_complete_solution(expanded_code, goal_description)
            
            # 3. Validate and Fallback if Needed
            complete_solution = self._validate_and_fallback(complete_solution, goal_description)
            
            # 4. Generate Documentation and Supporting Materials
            outputs.update(self._generate_documentation(expanded_code, complete_solution, goal_description))
            
            # 5. Generate Usage Examples
            outputs["usage_examples"] = self._generate_usage_examples(complete_solution, goal_description)
            
            # 6. Generate Code Analysis
            outputs["code_analysis"] = self._generate_code_analysis(complete_solution)
            
            outputs["complete_solution"] = complete_solution
            # Include standard outputs
            outputs.update(self._load_standard_outputs(goal, tools_used, execution_summary, results, raw_outputs))
            
            return outputs
            
        except Exception as e:
            self.log(f"Error generating coding-focused outputs: {str(e)}", "ERROR")
            return self._generate_comprehensive_outputs(goal, tools_used, execution_summary, results, raw_outputs)

    # Helper Methods --------------------------------------------------------------

    def _format_code_fragments(self, raw_outputs):
        """Extract and format code fragments from raw outputs"""
        code_results_formatted = ""
        for tool_name, result in raw_outputs.items():
            if tool_name in ["code_generator", "execute_python"]:
                code_results_formatted += f"\n--- {tool_name} OUTPUT ---\n"
                code_results_formatted += result if isinstance(result, str) else str(result)
        return code_results_formatted

    def _expand_code_fragments(self, code_fragments, goal_description):
        """Expand partial code fragments into complete implementations"""
        from langchain.prompts import PromptTemplate
        from langchain.schema.output_parser import StrOutputParser
        
        prompt = PromptTemplate(template="""
        Given these code fragments for '{goal_description}':
        
        {code_fragments}
        
        Expand into a complete implementation by:
        1. Filling all missing implementations
        2. Adding required imports
        3. Implementing stubs/todos
        4. Adding error handling
        5. Including necessary boilerplate
        6. Ensuring it can run independently
        
        Return only the expanded code in markdown blocks.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "code_fragments": code_fragments,
            "goal_description": goal_description
        })

    def _generate_complete_solution(self, expanded_code, goal_description):
        """Generate production-ready solution from expanded code"""
        from langchain.prompts import PromptTemplate
        from langchain.schema.output_parser import StrOutputParser
        
        prompt = PromptTemplate(template="""
        Refactor this code for '{goal_description}':
        
        {expanded_code}
        
        Create production-ready code that:
        1. Follows PEP 8 and type hints
        2. Has full error handling
        3. Includes docstrings
        4. Is modular and tested
        5. Includes configuration if needed
        6. Fully cover the use case
        
        Return:
        1. Main implementation
        2. Example tests
        3. requirements.txt (if needed)
        4. Usage examples in code comments
        
        It's Important the code fully cover the usecase.
        Format all components in markdown with language tags.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        solution = chain.invoke({
            "expanded_code": expanded_code,
            "goal_description": goal_description
        })
        return solution

    def _validate_and_fallback(self, solution, goal_description):
        """Ensure valid code output with fallback generation"""
        from langchain.prompts import PromptTemplate
        from langchain.schema.output_parser import StrOutputParser
        
        if not solution.strip():
            self.log("No code generated, attempting direct generation", "WARNING")
            prompt = PromptTemplate(template="""
            For: '{goal_description}'
            
            Generate complete Python implementation including:
            1. All required classes/functions
            2. Error handling
            3. Example usage
            4. Basic testing
            
            Return only the code in markdown blocks.
            """)
            
            chain = prompt | self.llm | StrOutputParser()
            solution = chain.invoke({"goal_description": goal_description})
        
        return solution

    def _generate_documentation(self, expanded_code, solution, goal_description):
        """Generate comprehensive documentation"""
        from langchain.prompts import PromptTemplate
        from langchain.schema.output_parser import StrOutputParser
        
        prompt = PromptTemplate(template="""
        For this code solution:
        
        {solution}
        
        And development artifacts:
        
        {expanded_code}
        
        Create full documentation for '{goal_description}' including:
        1. Architecture overview
        2. API reference
        3. Installation guide
        4. Usage examples
        5. Testing guide
        6. Deployment notes
        
        Format as Markdown with tabbed sections if needed.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        return {
            "documentation": chain.invoke({
                "solution": solution,
                "expanded_code": expanded_code,
                "goal_description": goal_description
            })
        }

    def _generate_usage_examples(self, solution, goal_description):
        """Generate executable usage examples"""
        from langchain.prompts import PromptTemplate
        from langchain.schema.output_parser import StrOutputParser
        
        prompt = PromptTemplate(template="""
        For this code:
        
        {solution}
        
        Create 5 practical usage examples showing:
        1. Different initialization configurations
        2. Common use cases
        3. Error handling scenarios
        4. Output processing
        5. Integration with other systems
        
        Make examples self-contained and executable.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "solution": solution[:10000],  # Increased size limit
            "goal_description": goal_description
        })

    def _generate_code_analysis(self, solution):
        """Generate technical analysis of the code"""
        from langchain.prompts import PromptTemplate
        from langchain.schema.output_parser import StrOutputParser
        
        prompt = PromptTemplate(template="""
        Analyze this code:
        
        {solution}
        
        Provide technical analysis covering:
        1. Code quality metrics
        2. Performance characteristics
        3. Security considerations
        4. Scalability assessment
        5. Improvement recommendations
        
        Use bullet points with specific code examples.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"solution": solution[:10000]})

    def _load_standard_outputs(self, *args):
        """Load standard outputs without overriding code-focused ones"""
        standard = self._generate_comprehensive_outputs(*args)
        return {k: v for k, v in standard.items() if k not in ["documentation", "code"]}

    def _generate_comprehensive_outputs(self, goal, tools_used, execution_summary, results, raw_outputs):
        """Generate comprehensive outputs in multiple formats based on tool execution results"""
        try:
            outputs = {}
            goal_description = goal.get("description", "No description provided")
            
            # 1. Generate executive summary
            prompt_exec_summary = PromptTemplate(template="""
            Based on the results of these tool executions:
            {results_summary}
            
            For this goal: {goal_description}
            
            Please provide a concise executive summary (3-5 paragraphs) that highlights:
            1. What was accomplished
            2. Key findings and insights
            3. Whether the goal was successfully completed
            4. Next steps or recommendations
            
            Format as Markdown with clear headings.
            """)
            
            # Format results summary for the prompt
            results_summary = ""
            for tool, result in results.items():
                results_summary += f"\n--- {tool} RESULT ---\n"
                result_text = result[:500] + "..." if isinstance(result, str) and len(result) > 500 else str(result)
                results_summary += result_text
            
            # Generate executive summary
            exec_summary_chain = prompt_exec_summary | self.llm | StrOutputParser()
            executive_summary = exec_summary_chain.invoke({
                "results_summary": results_summary,
                "goal_description": goal_description
            })
            outputs["executive_summary"] = executive_summary
            
            # 2. Generate detailed analysis
            prompt_detailed = PromptTemplate(template="""
            Based on the complete results of these tool executions:
            {full_results}
            
            For this goal: {goal_description}
            
            Please provide a comprehensive analysis that includes:
            1. Detailed breakdown of each tool's findings
            2. Patterns, insights, and connections between different results
            3. Critical analysis of the information gathered
            4. Limitations of the current approach and potential blind spots
            5. Actionable recommendations with justifications
            
            Format as a professional Markdown document with clear sections, bullet points, and emphasis where appropriate.
            """)
            
            # Format full results for the prompt
            full_results = ""
            for tool, result in raw_outputs.items():
                full_results += f"\n\n## {tool.upper()} RESULTS\n\n"
                result_text = result if isinstance(result, str) else str(result)
                full_results += result_text
            
            # Generate detailed analysis
            detailed_chain = prompt_detailed | self.llm | StrOutputParser()
            detailed_analysis = detailed_chain.invoke({
                "full_results": full_results,
                "goal_description": goal_description
            })
            outputs["detailed_analysis"] = detailed_analysis
            
            # 3. Generate visualizations summary (instruction for visualization creation)
            prompt_visuals = PromptTemplate(template="""
            Based on the results of these tool executions:
            {results_summary}
            
            For this goal: {goal_description}
            
            Identify what data from the results would be valuable to visualize.
            For each visualization opportunity:
            1. Describe what data should be visualized
            2. Suggest the appropriate visualization type (bar chart, line graph, etc.)
            3. Explain what insights this visualization would reveal
            
            Format as Markdown with a clear section for each recommended visualization.
            """)
            
            # Generate visualization recommendations
            visuals_chain = prompt_visuals | self.llm | StrOutputParser()
            visualization_summary = visuals_chain.invoke({
                "results_summary": results_summary,
                "goal_description": goal_description
            })
            outputs["visualization_recommendations"] = visualization_summary
            
            # 4. Generate action plan
            prompt_action = PromptTemplate(template="""
            Based on the analysis of the tool execution results:
            {executive_summary}
            
            For this goal: {goal_description}
            
            Please provide a specific, actionable plan that includes:
            1. Clear next steps with priorities
            2. Required resources or information for each step
            3. Potential challenges and how to overcome them
            4. Success metrics to evaluate progress
            
            Format as a structured action plan in Markdown with numbered steps and clear headings.
            """)
            
            # Generate action plan
            action_chain = prompt_action | self.llm | StrOutputParser()
            action_plan = action_chain.invoke({
                "executive_summary": executive_summary,
                "goal_description": goal_description
            })
            outputs["action_plan"] = action_plan
            
            # 5. Generate exportable data in structured format if applicable
            # Look for data that could be structured (e.g., from web scraping, system info)
            has_structured_data = any(tool in raw_outputs for tool in ["web_scrape", "system_info", "read_directory"])
            if has_structured_data:
                prompt_structured = PromptTemplate(template="""
                From these tool execution results:
                {full_results}
                
                Extract all structured data into a well-formatted JSON structure.
                Focus on organizing key-value pairs, lists, and nested data in a clean, consistent format.
                Include only the actual data, not explanations or context.
                """)
                
                # Generate structured data
                structured_chain = prompt_structured | self.llm | StrOutputParser()
                try:
                    structured_data = structured_chain.invoke({
                        "full_results": full_results
                    })
                    
                    # Try to parse and validate as JSON
                    try:
                        json_data = json.loads(structured_data)
                        # If successful, convert back to pretty JSON string
                        outputs["structured_data"] = json.dumps(json_data, indent=2)
                    except:
                        # If failed, still store the output but mark as unvalidated
                        outputs["structured_data"] = structured_data
                except Exception as e:
                    self.log(f"Error generating structured data: {str(e)}", "WARNING")
            
            return outputs
        
        except Exception as e:
            self.log(f"Error generating comprehensive outputs: {str(e)}", "ERROR")
            return {"error": str(e)}

    def _create_execution_documentation(self, goal, tools_used, execution_summary, results, outputs=None):
        """Create detailed documentation of the goal execution with proper encoding"""
        try:
            goal_description = goal.get("description", "No description provided")
            goal_id = goal.get("id", 0)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            date_simple = datetime.now().strftime('%Y%m%d')
            
            # Create a uniquely named folder for this goal's outputs
            doc_folder = f"goal_{goal_id}_{date_simple}"
            os.makedirs(f"./workspace/{doc_folder}", exist_ok=True)
            
            # Replace Unicode emoji with ASCII alternatives to avoid encoding issues
            success_mark = "✓" # Replace with "(Success)" if encoding issues persist
            failure_mark = "✗" # Replace with "(Failed)" if encoding issues persist

            # 1. Main markdown report with comprehensive info
            doc_content = f"""# Goal Execution Report: {goal_description}

    ## Overview
    - **Goal ID:** {goal_id}
    - **Status:** Completed
    - **Execution Time:** {timestamp}
    - **Tools Used:** {', '.join(tools_used)}

    ## Executive Summary
    {outputs.get('executive_summary', 'No summary available.')}

    ## Execution Details
    """
            
            # Add detailed execution information with enhanced formatting
            for step in execution_summary:
                tool_name = step['tool']
                success = f"{success_mark} Success" if step.get('success', False) else f"{failure_mark} Failed"
                exec_time = f"{step.get('execution_time', 0):.2f}s"
                
                doc_content += f"{NEW_LINE}### {tool_name} ({success}, {exec_time}){NEW_LINE}"
                
                # Add input used
                doc_content += "\n**Input:**\n```json\n"
                doc_content += json.dumps(step.get('input', {}), indent=2)
                doc_content += "\n```\n"
                
                # Add result or error
                if step.get('success', False):
                    doc_content += "\n**Result Preview:**\n```\n"
                    result_preview = step.get('result_snippet', 'No result')
                    doc_content += result_preview
                    doc_content += "\n```\n"
                    
                    # Add link to full result file if result is large
                    if tool_name != "web_scrape":
                        result = results.get(tool_name, '')
                        if isinstance(result, str) and len(result) > 1000:
                            # Save full result to separate file
                            result_filename = f"{tool_name.lower().replace(' ', '_')}_result.txt"
                            with open(f"./workspace/{doc_folder}/{result_filename}", "w", encoding="utf-8") as f:
                                f.write(result)
                            doc_content += f"{NEW_LINE}[View Full Result](./{result_filename}){NEW_LINE}"
                else:
                    doc_content += f"{NEW_LINE}**Error:** {step.get('error', 'Unknown error')}{NEW_LINE}"
            
            # Add code documentation section if available
            if outputs and 'documentation' in outputs:
                doc_content += "\n## Code Documentation\n"
                doc_content += outputs['documentation']
                
                # Save complete solution to separate file if available
                if 'complete_solution' in outputs and outputs['complete_solution']:
                    solution_filename = f"goal_{goal_id}_solution"
                    with open(f"./workspace/{doc_folder}/{solution_filename}", "w", encoding="utf-8") as f:
                        f.write(outputs['complete_solution'])
                    doc_content += f"\n[View Complete Solution](./{solution_filename})\n"
                    
                    # Save usage examples if available
                    if 'usage_examples' in outputs and outputs['usage_examples']:
                        examples_filename = f"goal_{goal_id}_examples"
                        with open(f"./workspace/{doc_folder}/{examples_filename}", "w", encoding="utf-8") as f:
                            f.write(outputs['usage_examples'])
                        doc_content += f"\n[View Usage Examples](./{examples_filename})\n"
            
            # Add detailed analysis
            if outputs and 'detailed_analysis' in outputs:
                doc_content += "\n## Detailed Analysis\n"
                doc_content += outputs['detailed_analysis']
            
            # Add code analysis if available
            if outputs and 'code_analysis' in outputs:
                doc_content += "\n## Code Quality Analysis\n"
                doc_content += outputs['code_analysis']
            
            # Add visualization recommendations
            if outputs and 'visualization_recommendations' in outputs:
                doc_content += "\n## Visualization Opportunities\n"
                doc_content += outputs['visualization_recommendations']
            
            # Add action plan
            if outputs and 'action_plan' in outputs:
                doc_content += "\n## Recommended Action Plan\n"
                doc_content += outputs['action_plan']
                
            # Save the main documentation file - ADD UTF-8 ENCODING HERE
            main_doc_filename = f"goal_{goal_id}_{date_simple}_report.md"
            with open(f"./workspace/{doc_folder}/{main_doc_filename}", "w", encoding="utf-8") as f:
                f.write(doc_content)
                
            # 2. Create an HTML report for interactive viewing
            # Update your template to include the marked.js library and implement markdown rendering
            styles = """
                <style>
                    :root {
                        --primary-color: #3498db;
                        --primary-dark: #2980b9;
                        --secondary-color: #2c3e50;
                        --accent-color: #16a085;
                        --success-color: #2ecc71;
                        --warning-color: #f39c12;
                        --danger-color: #e74c3c;
                        --light-bg: #f8f9fa;
                        --dark-bg: #2c3e50;
                        --text-color: #333;
                        --text-light: #6c757d;
                        --text-lighter: #adb5bd;
                        --border-color: #e0e0e0;
                        --box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        --transition: all 0.3s ease;
                        --border-radius: 8px;
                    }

                    body {
                        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                        line-height: 1.7;
                        color: var(--text-color);
                        max-width: 1280px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #fff;
                    }

                    header {
                        background: linear-gradient(135deg, var(--dark-bg), var(--primary-dark));
                        color: white;
                        padding: 30px;
                        border-radius: var(--border-radius);
                        margin-bottom: 30px;
                        box-shadow: var(--box-shadow);
                        position: relative;
                        overflow: hidden;
                    }

                    header::after {
                        content: '';
                        position: absolute;
                        top: 0;
                        right: 0;
                        bottom: 0;
                        left: 0;
                        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0));
                        pointer-events: none;
                    }

                    h1, h2, h3, h4 {
                        color: var(--secondary-color);
                        margin-top: 1.5em;
                        margin-bottom: 0.7em;
                        font-weight: 600;
                        line-height: 1.3;
                    }

                    h1 {
                        font-size: 2.2rem;
                        margin-top: 0;
                    }

                    h2 {
                        font-size: 1.8rem;
                    }

                    h3 {
                        font-size: 1.4rem;
                    }

                    h4 {
                        font-size: 1.2rem;
                    }

                    .tool-card {
                        border: 1px solid var(--border-color);
                        border-radius: var(--border-radius);
                        margin-bottom: 24px;
                        overflow: hidden;
                        transition: var(--transition);
                        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                    }

                    .tool-card:hover {
                        box-shadow: var(--box-shadow);
                        transform: translateY(-2px);
                    }

                    .tool-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 14px 20px;
                        background-color: #f5f7fa;
                        border-bottom: 1px solid var(--border-color);
                    }

                    .tool-content {
                        padding: 20px;
                    }

                    .success {
                        color: var(--success-color);
                        font-weight: 500;
                    }

                    .warning {
                        color: var(--warning-color);
                        font-weight: 500;
                    }

                    .failure {
                        color: var(--danger-color);
                        font-weight: 500;
                    }

                    pre {
                        background-color: #f8f9fa;
                        border: 1px solid var(--border-color);
                        border-radius: 6px;
                        padding: 16px;
                        overflow-x: auto;
                        font-family: 'Fira Code', 'JetBrains Mono', Menlo, Consolas, Monaco, monospace;
                        font-size: 0.9rem;
                        line-height: 1.6;
                    }

                    code {
                        font-family: 'Fira Code', 'JetBrains Mono', Menlo, Consolas, Monaco, monospace;
                        font-size: 0.9em;
                        background-color: rgba(27,31,35,0.05);
                        border-radius: 3px;
                        padding: 0.2em 0.4em;
                    }

                    .executive-summary {
                        background-color: #f8fafc;
                        border-left: 4px solid var(--primary-color);
                        padding: 20px;
                        margin-bottom: 25px;
                        border-radius: 0 var(--border-radius) var(--border-radius) 0;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                    }

                    .action-plan {
                        background-color: #f0f8f7;
                        border-left: 4px solid var(--accent-color);
                        padding: 20px;
                        margin: 25px 0;
                        border-radius: 0 var(--border-radius) var(--border-radius) 0;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                    }

                    /* Modern tabs */
                    .tab {
                        display: flex;
                        overflow: hidden;
                        border: none;
                        background-color: transparent;
                        border-radius: var(--border-radius) var(--border-radius) 0 0;
                        margin-bottom: -1px;
                    }

                    .tab button {
                        background-color: #f5f7fa;
                        border: 1px solid var(--border-color);
                        border-bottom: none;
                        border-radius: var(--border-radius) var(--border-radius) 0 0;
                        outline: none;
                        cursor: pointer;
                        padding: 12px 18px;
                        margin-right: 6px;
                        transition: var(--transition);
                        font-weight: 500;
                        color: var(--text-light);
                    }

                    .tab button:hover {
                        background-color: #fff;
                        color: var(--primary-color);
                    }

                    .tab button.active {
                        background-color: #fff;
                        color: var(--primary-color);
                        border-bottom: 2px solid var(--primary-color);
                    }

                    .tabcontent {
                        display: none;
                        padding: 24px;
                        border: 1px solid var(--border-color);
                        border-radius: 0 var(--border-radius) var(--border-radius) var(--border-radius);
                        background-color: #fff;
                        animation: fadeIn 0.3s ease;
                    }

                    @keyframes fadeIn {
                        from { opacity: 0; }
                        to { opacity: 1; }
                    }

                    .code-section {
                        background-color: #f8f9fa;
                        border: 1px solid var(--border-color);
                        border-radius: var(--border-radius);
                        padding: 16px;
                        margin: 20px 0;
                    }

                    .code-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 12px;
                        padding-bottom: 10px;
                        border-bottom: 1px solid var(--border-color);
                        font-weight: 500;
                    }

                    /* Markdown styling improvements */
                    .markdown-content {
                        line-height: 1.8;
                    }

                    .markdown-content h1 {
                        font-size: 2rem;
                        border-bottom: 1px solid var(--border-color);
                        padding-bottom: 0.5em;
                        margin-bottom: 1em;
                    }

                    .markdown-content h2 {
                        font-size: 1.6rem;
                        border-bottom: 1px solid var(--border-color);
                        padding-bottom: 0.4em;
                        margin-bottom: 0.8em;
                    }

                    .markdown-content h3 {
                        font-size: 1.3rem;
                        margin-bottom: 0.7em;
                    }

                    .markdown-content h4 {
                        font-size: 1.1rem;
                        margin-bottom: 0.6em;
                    }

                    .markdown-content blockquote {
                        border-left: 4px solid var(--primary-color);
                        color: var(--text-light);
                        margin: 1em 0;
                        padding: 0.5em 1.2em;
                        background-color: rgba(52, 152, 219, 0.05);
                        border-radius: 0 var(--border-radius) var(--border-radius) 0;
                    }

                    .markdown-content table {
                        border-collapse: collapse;
                        width: 100%;
                        margin: 1.5em 0;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
                        border-radius: var(--border-radius);
                        overflow: hidden;
                    }

                    .markdown-content table th {
                        background-color: #f5f7fa;
                        font-weight: 600;
                        text-align: left;
                        padding: 12px 15px;
                        border: 1px solid var(--border-color);
                    }

                    .markdown-content table td {
                        padding: 10px 15px;
                        border: 1px solid var(--border-color);
                    }

                    .markdown-content table tr:nth-child(2n) {
                        background-color: #f8f9fa;
                    }

                    .markdown-content table tr:hover {
                        background-color: rgba(52, 152, 219, 0.05);
                    }

                    /* Button styles */
                    .btn {
                        display: inline-block;
                        padding: 10px 16px;
                        border-radius: var(--border-radius);
                        font-weight: 500;
                        text-decoration: none;
                        cursor: pointer;
                        transition: var(--transition);
                        border: none;
                        text-align: center;
                    }

                    .btn-primary {
                        background-color: var(--primary-color);
                        color: white;
                    }

                    .btn-primary:hover {
                        background-color: var(--primary-dark);
                        transform: translateY(-1px);
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    }

                    .btn-secondary {
                        background-color: #f5f7fa;
                        color: var(--text-color);
                        border: 1px solid var(--border-color);
                    }

                    .btn-secondary:hover {
                        background-color: #e9ecef;
                    }

                    /* Card grid layout */
                    .card-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                        gap: 24px;
                        margin: 20px 0;
                    }

                    /* Responsive design */
                    @media (max-width: 768px) {
                        header {
                            padding: 20px;
                        }
                        
                        h1 {
                            font-size: 1.8rem;
                        }
                        
                        h2 {
                            font-size: 1.5rem;
                        }
                        
                        .card-grid {
                            grid-template-columns: 1fr;
                        }
                        
                        .tab button {
                            padding: 10px 14px;
                            font-size: 0.9rem;
                        }
                    }
                </style>
            """
            html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Goal Execution Report: {goal_description}</title>
                    <!-- Include the marked.js library from CDN -->
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.2.1/marked.min.js"></script>
                </head>
                <body>
                    {styles}
                    <header>
                        <h1>Goal Execution Report</h1>
                        <p>{goal_description}</p>
                        <div>
                            <strong>Goal ID:</strong> {goal_id} | 
                            <strong>Execution:</strong> {timestamp} |
                            <strong>Tools:</strong> {', '.join(tools_used)}
                        </div>
                    </header>
                    
                    <div class="tab">
                        <button class="tablinks active" onclick="openTab(event, 'Summary')">Executive Summary</button>
                        <button class="tablinks" onclick="openTab(event, 'Execution')">Execution Details</button>
                        <button class="tablinks" onclick="openTab(event, 'Code')">Code Documentation</button>
                        <button class="tablinks" onclick="openTab(event, 'Analysis')">Analysis</button>
                        <button class="tablinks" onclick="openTab(event, 'Action')">Action Plan</button>
                    </div>
                    
                    <div id="Summary" class="tabcontent" style="display: block;">
                        <div class="executive-summary markdown-content" id="executive-summary-content">
                            {outputs.get('executive_summary', 'No summary available.')}
                        </div>
                    </div>
                    
                    <div id="Execution" class="tabcontent">
                        <h2>Tool Execution Details</h2>
                """
                        
            # Add execution details
            for step in execution_summary:
                tool_name = step['tool']
                success = step.get('success', False)
                exec_time = step.get('execution_time', 0)
                status_class = "success" if success else "failure"
                status_icon = success_mark if success else failure_mark
                
                html_content += f"""
            <div class="tool-card">
                <div class="tool-header">
                    <h3>{tool_name}</h3>
                    <span class="{status_class}">{status_icon} {exec_time:.2f}s</span>
                </div>
                <div class="tool-content">
                    <h4>Input:</h4>
                    <pre>{json.dumps(step.get('input', {}), indent=2)}</pre>
                """
                
                if success:
                    html_content += f"""
                    <h4>Result Preview:</h4>
                    <pre>{step.get('result_snippet', 'No result')}</pre>
                    """
                    
                    # Add link to full result file if result is large
                    result = results.get(tool_name, '')
                    if isinstance(result, str) and len(result) > 1000:
                        result_filename = f"{tool_name.lower().replace(' ', '_')}_result.txt"
                        html_content += f"""
                    <p><a href="{result_filename}" target="_blank">View Full Result</a></p>
                        """
                else:
                    html_content += f"""
                    <h4>Error:</h4>
                    <pre class="failure">{step.get('error', 'Unknown error')}</pre>
                    """
                    
                html_content += """
                </div>
            </div>
                """
                
            # Add code documentation tab if available
            html_content += """
        </div>
        """
            
            if outputs and 'documentation' in outputs:
                html_content += f"""
        <div id="Code" class="tabcontent">
            <h2>Code Documentation</h2>
            <div class="code-section markdown-content" id="documentation-content">
                {outputs.get('documentation', '')}
            </div>
                """
                
                if 'complete_solution' in outputs and outputs['complete_solution']:
                    solution_filename = f"goal_{goal_id}_solution.py"
                    html_content += f"""
            <div class="code-header">
                <h3>Complete Solution</h3>
                <a href="{solution_filename}" download>Download Solution</a>
            </div>
            <div class="markdown-content" id="complete-solution-content">
                {outputs.get('complete_solution', '')}
            </div>
            <p><a href="{solution_filename}">View Full Solution</a></p>
                    """
                    
                if 'usage_examples' in outputs and outputs['usage_examples']:
                    examples_filename = f"goal_{goal_id}_examples.py"
                    html_content += f"""
            <div class="code-header">
                <h3>Usage Examples</h3>
                <a href="{examples_filename}" download>Download Examples</a>
            </div>
            <div class="markdown-content" id="usage-examples-content">
                {outputs.get('usage_examples', '')}
            </div>
            <p><a href="{examples_filename}">View Full Examples</a></p>
                    """
                    
                if 'code_analysis' in outputs and outputs['code_analysis']:
                    html_content += f"""
            <h3>Code Quality Analysis</h3>
            <div class="code-section markdown-content" id="code-analysis-content">
                {outputs.get('code_analysis', '')}
            </div>
                    """
                    
                html_content += """
        </div>
                """
            functions = """
                 function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}
        
        // Parse all markdown content on page load
        document.addEventListener('DOMContentLoaded', function() {
            // Get all elements with markdown content
            const markdownElements = document.querySelectorAll('.markdown-content');
            
            // Render markdown content for each element
            markdownElements.forEach(function(element) {
                const rawContent = element.textContent || element.innerText;
                element.innerHTML = marked.parse(rawContent);
            });
        });
            """
            # Add analysis tab content
            html_content += f"""
        <div id="Analysis" class="tabcontent">
            <h2>Detailed Analysis</h2>
            <div class="markdown-content" id="detailed-analysis-content">
                {outputs.get('detailed_analysis', 'No analysis available.')}
            </div>
            
            <h2>Visualization Opportunities</h2>
            <div class="markdown-content" id="visualization-recommendations-content">
                {outputs.get('visualization_recommendations', 'No recommendations available.')}
            </div>
        </div>
        
        <div id="Action" class="tabcontent">
            <h2>Recommended Action Plan</h2>
            <div class="action-plan markdown-content" id="action-plan-content">
                {outputs.get('action_plan', 'No action plan available.')}
            </div>
        </div>
        
        <script>
            {functions}
        </script>
    </body>
    </html>
            """
            
            # Save the HTML report - ADD UTF-8 ENCODING HERE
            html_filename = f"goal_{goal_id}_{date_simple}_report.html"
            with open(f"./workspace/{doc_folder}/{html_filename}", "w", encoding="utf-8") as f:
                f.write(html_content)
                
            # 3. Save structured data as JSON if available
            if outputs and 'structured_data' in outputs:
                json_filename = f"goal_{goal_id}_{date_simple}_data.json"
                with open(f"./workspace/{doc_folder}/{json_filename}", "w", encoding="utf-8") as f:
                    f.write(outputs['structured_data'])
            
            # 4. Create a simple README to guide users through the outputs
            readme_content = f"""# Goal Execution: {goal_description}

    ## Available Files
    - **[{main_doc_filename}](./{main_doc_filename})** - Comprehensive Markdown report with all details
    - **[{html_filename}](./{html_filename})** - Interactive HTML report with tabbed interface

    ## Quick Access
    - For a quick overview: Open the HTML report and check the Executive Summary tab
    - For technical details: See the Execution Details tab in the HTML report
    - For code documentation: Check the Code Documentation tab
    - For recommended next steps: Check the Action Plan tab

    ## Additional Resources
    """
            # Add links to any extra files
            if outputs and 'structured_data' in outputs:
                readme_content += f"- **[{json_filename}](./{json_filename})** - Structured data in JSON format{NEW_LINE}"
                
            # Add code solution files if available
            if outputs and 'complete_solution' in outputs and outputs['complete_solution']:
                solution_filename = f"goal_{goal_id}_solution.py"
                readme_content += f"- **[{solution_filename}](./{solution_filename})** - Complete implementation{NEW_LINE}"
                
            if outputs and 'usage_examples' in outputs and outputs['usage_examples']:
                examples_filename = f"goal_{goal_id}_examples.py"
                readme_content += f"- **[{examples_filename}](./{examples_filename})** - Usage examples{NEW_LINE}"
                
            for tool_name in results:
                result = results.get(tool_name, '')
                if isinstance(result, str) and len(result) > 1000:
                    result_filename = f"{tool_name.lower().replace(' ', '_')}_result.txt"
                    readme_content += f"- **[{result_filename}](./{result_filename})** - Full results from {tool_name}{NEW_LINE}"
            
            # Save the README - ADD UTF-8 ENCODING HERE
            with open(f"./workspace/{doc_folder}/README.md", "w", encoding="utf-8") as f:
                f.write(readme_content)
            
            # Log success
            self.log(f"Created comprehensive execution documentation in workspace/{doc_folder}/", "INFO")
            return f"workspace/{doc_folder}/"
            
        except Exception as e:
            self.log(f"Error creating execution documentation: {str(e)}", "ERROR")
            return None

    def _store_goal_results(self, goal, tools_used, results, execution_summary, outputs=None):
        """Store goal execution results in vector memory with enhanced metadata for future reference"""
        try:
            # Create a combined result text for embedding
            combined_text = f"Goal: {goal['description']}{NEW_LINE}{NEW_LINE}Tools used: {', '.join(tools_used)}{NEW_LINE}{NEW_LINE}"
            
            # Add executive summary if available
            if outputs and 'executive_summary' in outputs:
                combined_text += f"Executive Summary:{NEW_LINE}{outputs['executive_summary']}{NEW_LINE}{NEW_LINE}"
            
            # Add tool results
            for tool, result in results.items():
                result_text = result[:1000] + "..." if isinstance(result, str) and len(result) > 1000 else str(result)
                combined_text += f"--- {tool} RESULT ---{NEW_LINE}{result_text}{NEW_LINE}{NEW_LINE}"
            
            # Add action plan if available
            if outputs and 'action_plan' in outputs:
                combined_text += f"Action Plan:{NEW_LINE}{outputs['action_plan']}{NEW_LINE}{NEW_LINE}"
            
            # Create embeddings
            embeddings = self.embedding_function([combined_text])
            
            # Generate a unique ID
            result_id = f"goal_result_{goal['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create metadata
            # metadata = {
            #     "type": "goal_execution",
            #     "goal_id": goal['id'],
            #     "goal_description": goal['description'],
            #     "tools_used": tools_used,
            #     "timestamp": datetime.now().isoformat(),
            #     "success": True,  # Assuming success if we get here
            #     "output_types": list(outputs.keys()) if outputs else []
            # }
            
            # Store in vector database
            self.core_collection.add(
                ids=[result_id],
                embeddings=embeddings,
                # metadatas=[metadata]
            )
            
            # Add to execution history in memory with enhanced tracking
            execution_record = {
                "goal_id": goal['id'],
                "timestamp": datetime.now().isoformat(),
                "tools_used": tools_used,
                "summary": execution_summary,
                "result_id": result_id,
                # "output_location": metadata.get("output_location", ""),
                "performance_metrics": {
                    "total_execution_time": sum(step.get("execution_time", 0) for step in execution_summary),
                    "tool_count": len(tools_used),
                    "success_rate": sum(1 for step in execution_summary if step.get("success", False)) / len(execution_summary) if execution_summary else 0
                }
            }
            
            self.agent_memory['execution_history'].append(execution_record)
            
            # Update goal statistics
            if 'goal_statistics' not in self.agent_memory:
                self.agent_memory['goal_statistics'] = {}
                
            self.agent_memory['goal_statistics'][goal['id']] = {
                "last_execution": datetime.now().isoformat(),
                "execution_count": self.agent_memory['goal_statistics'].get(goal['id'], {}).get("execution_count", 0) + 1,
                "tools_used": tools_used,
                "success": True,
                "result_id": result_id
            }
            
            self.log(f"Stored comprehensive goal execution results: {result_id}", "INFO")
            return result_id
            
        except Exception as e:
            self.log(f"Error storing goal results: {str(e)}", "ERROR")
            return None
        
    def reflect_on_progress(self, completed_goal, success):
        """Use LangChain for reflection and planning"""
        # First, retrieve relevant memory from vector store
        query = f"{self.objective} {completed_goal['description']}"
        embeddings = self.embedding_function([query])
        relevant_docs = self.core_collection.query(embeddings)
        relevant_memory_text = "\n".join([doc for doc in relevant_docs])
        
        # Get the remaining pending goals
        pending_goals = [g for g in self.goals if g['status'] == 'PENDING']
        completed_goals = [g for g in self.goals if g['status'] == 'COMPLETED']
        
        prompt = PromptTemplate(template="""
            Current Objective: {objective}
            Recently completed goal: {completed_goal}
            Success: {success}
            Human Feedback: {human_feedback}
            Relevant Memory: {memory}
            Pending Goals: {pending_goals}
            Completed Goals: {completed_goals}
            
            Based on this information, perform a thorough analysis:
            
            1. Progress Assessment: How much progress has been made toward the overall objective? (0-100%)
            2. Gap Analysis: What specific aspects of the objective remain unaddressed?
            3. Next Steps: What specific, concrete actions are now needed to advance toward the objective?
            
            Choose exactly ONE action:
            1. Create new sub-goals (specify 1-3 clear, measurable goals to address gaps) of the original goal
            2. Create a new tool (only if a specific capability gap exists)
            3. Mark objective as complete (ONLY if truly 100% achieved and verified)
            4. Continue with existing goals (if they're sufficient to complete the objective)
            5. Revesit an existing goal (if it needs to be re-evaluated)
            6. Add more sub goals
            Return a JSON object with format:
            {{"action":"create_goals"|"create_tool"|"complete"|"continue",
            "reasoning":"Detailed explanation for your decision including progress assessment",
            "progress_percentage": number,
            "remaining_gaps": ["specific gap 1", "specific gap 2"],
            "details":[]|"tool description"|"completion message"|null}}
            
            BE AMBITIOUS BUT REALISTIC. Consider human feedback carefully when deciding the next action.
        """)
                
        # Create a runnable chain
        reflect_chain = prompt | self.llm | StrOutputParser()
        
        try:
            # Invoke the chain
            reflection = reflect_chain.invoke({
                "objective": self.objective,
                "completed_goal": json.dumps(completed_goal),
                "success": success,
                "human_feedback": self.human_feedback if self.human_feedback else "No specific feedback provided.",
                "memory": relevant_memory_text,
                "pending_goals": json.dumps(pending_goals, indent=2),
                "completed_goals": json.dumps(completed_goals, indent=2)
            })
            
            # Extract JSON from the reflection
            json_match = re.search(r'\{.*\}', reflection.replace('\n', ' '), re.DOTALL)
            if json_match:
                reflection_json = json_match.group()
            else:
                reflection_json = reflection
            
            self.agent_memory['reflections'].append({
                "timestamp": datetime.now().isoformat(),
                "analysis": reflection_json,
                "completed_goal": completed_goal["description"],
                "human_feedback": self.human_feedback
            })
            
            self.log(f"Reflection: {reflection_json[:200]}..." if len(reflection_json) > 200 else reflection_json)
            self._process_reflection(reflection_json)
        except Exception as e:
            self.log_error(f"Reflection failed: {str(e)}")
            # Fallback strategy
            if len(pending_goals) == 0 and len(completed_goals) > 0:
                self._add_goal(f"Verify objective completion", 1)

    def _process_reflection(self, reflection):
        """Process the LLM's reflection output"""
        try:
            # Try parsing as JSON
            decision = json.loads(reflection)
            action = decision.get('action')
            details = decision.get('details')
            reasoning = decision.get('reasoning', '')
            
            self.log(f"Decision: {action}, Reasoning: {reasoning[:100]}..." if len(reasoning) > 100 else reasoning)
            
            if action == 'create_goals':
                # Only create new goals if there are no pending goals or if specifically instructed by human
                pending_goals = [g for g in self.goals if g['status'] == 'PENDING']
                if len(pending_goals) == 0 or "new goal" in self.human_feedback.lower():
                    if isinstance(details, list):
                        # Limit to 1-2 new goals at a time
                        for i, goal_desc in enumerate(details[:2]):
                            self._add_goal(goal_desc, i+1)
                    else:
                        self._add_goal(str(details), 1)
                else:
                    self.log("Not creating new goals since there are still pending goals")
                    
            elif action == 'create_tool':
                tool_desc = details if isinstance(details, str) else decision.get('reasoning', 'Custom tool')
                self.create_tool(tool_desc)
                
            elif action == 'complete':
                # Only mark as complete if there are no pending goals
                pending_goals = [g for g in self.goals if g['status'] == 'PENDING']
                if not pending_goals:
                    self.agent_memory['system']['status'] = 'COMPLETED'
                    self.log("Objective marked as complete")
                else:
                    self.log("Cannot mark as complete - there are still pending goals")
            
            # Default action 'continue' doesn't need specific handling
            
        except json.JSONDecodeError:
            self.log_error("Failed to parse reflection as JSON")
            # Extract action from text if JSON parsing fails
            if "create_goals" in reflection.lower() and (len([g for g in self.goals if g['status'] == 'PENDING']) == 0):
                self._add_goal("Continue progress toward objective", 1)
            elif "create_tool" in reflection.lower():
                self.create_tool("Tool to further objective completion")
            elif "complete" in reflection.lower() and "objective" in reflection.lower():
                if len([g for g in self.goals if g['status'] == 'PENDING']) == 0:
                    self.agent_memory['system']['status'] = 'COMPLETED'
        except Exception as e:
            self.log_error(f"Failed to process reflection: {str(e)}")


    def start_monitoring_dashboard(self):
        """Start a web server with real-time monitoring and visualization dashboard"""
        import threading
        import http.server
        import socketserver
        from io import BytesIO
        import base64
        import matplotlib.pyplot as plt
        import numpy as np
        
        class AgentDashboardHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                return 
            
            def do_GET(self):
                path = self.path.split('?')[0]
                if path == '/':
                    # Serve the main dashboard page
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    # Generate HTML for dashboard
                    html = self.generate_dashboard_html()
                    self.wfile.write(html.encode())
                    
                elif path == '/status':
                    # Serve JSON status for API clients
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Cache-Control', 'no-cache, no-store')
                    self.end_headers()
                    
                    status_data = self.get_status_data()
                    self.wfile.write(json.dumps(status_data).encode())
                    
                elif path == '/logs':
                    # Serve recent logs
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.send_header('Cache-Control', 'no-cache, no-store')
                    self.end_headers()
                    
                    logs = self.get_recent_logs()
                    self.wfile.write(logs.encode())
            
            def get_status_data(self):
                """Get current agent status data"""
                agent = self.server.agent
                
                # Calculate goal statistics
                goal_stats = {"COMPLETED": 0, "FAILED": 0, "PENDING": 0}
                for goal in agent.goals:
                    goal_stats[goal["status"]] += 1
                    
                # Get most recent logs
                recent_logs = self.get_recent_logs(10)
                
                return {
                    "objective": agent.objective,
                    "status": agent.agent_memory['system']['status'],
                    "iteration": agent.current_iteration,
                    "max_iterations": agent.max_iterations,
                    "goals": {
                        "total": len(agent.goals),
                        "statistics": goal_stats,
                        "details": agent.goals
                    },
                    "tools": {
                        "base_tools": [t.name for t in agent.base_tools],
                        "custom_tools": list(agent.custom_tools.keys())
                    },
                    "recent_logs": recent_logs.split('\n')[-10:] if recent_logs else [],
                    "human_feedback": agent.human_feedback,
                    "last_updated": datetime.now().isoformat()
                }
            
            def get_recent_logs(self, limit=50):
                """Get the most recent log entries"""
                log_dir = os.path.join("./workspace", "logs")
                if not os.path.exists(log_dir):
                    return "No logs available"
                    
                # Get latest log file
                log_files = sorted(glob.glob(os.path.join(log_dir, "agent_log_*.txt")))
                if not log_files:
                    return "No log files found"
                    
                latest_log = log_files[-1]
                try:
                    with open(latest_log, 'r') as f:
                        lines = f.readlines()
                        return ''.join(lines[-limit:])
                except Exception as e:
                    return f"Error reading logs: {str(e)}"
            
            def generate_dashboard_html(self):
                """Generate HTML for the dashboard with auto-refresh"""
                html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Agent Monitoring Dashboard</title>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                        .container { max-width: 1200px; margin: 0 auto; }
                        .header { background-color: #333; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                        .card { background-color: white; border-radius: 5px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                        .flex-container { display: flex; flex-wrap: wrap; gap: 20px; }
                        .flex-item { flex: 1; min-width: 300px; }
                        .status-badge { display: inline-block; padding: 5px 10px; border-radius: 3px; color: white; }
                        .status-in-progress { background-color: #007bff; }
                        .status-completed { background-color: #28a745; }
                        .status-max-reached { background-color: #dc3545; }
                        table { width: 100%; border-collapse: collapse; }
                        table, th, td { border: 1px solid #ddd; }
                        th, td { padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                        tr:nth-child(even) { background-color: #f9f9f9; }
                        .log-container { height: 300px; overflow-y: scroll; background-color: #f8f8f8; padding: 10px; border: 1px solid #ddd; font-family: monospace; }
                        .log-item { margin: 0; padding: 2px 0; border-bottom: 1px solid #eee; }
                        .log-item:last-child { border-bottom: none; }
                        .viz-container { text-align: center; }
                        .viz-container img { max-width: 100%; height: auto; }
                    </style>
                    <script>
                        // Auto-refresh components
                        function refreshData() {
                            fetch('/status')
                                .then(response => response.json())
                                .then(data => updateDashboard(data))
                                .catch(error => console.error('Error fetching status:', error));
                                
                            // Refresh logs
                            fetch('/logs')
                                .then(response => response.text())
                                .then(data => {
                                    const logContainer = document.getElementById('log-container');
                                    const logs = data.split('\\n');
                                    let logHtml = '';
                                    logs.forEach(log => {
                                        if (log.trim()) {
                                            logHtml += `<p class="log-item">${log}</p>`;
                                        }
                                    });
                                    logContainer.innerHTML = logHtml;
                                    logContainer.scrollTop = logContainer.scrollHeight;
                                })
                                .catch(error => console.error('Error fetching logs:', error));
                        }
                        
                        function updateDashboard(data) {
                            // Update header stats
                            document.getElementById('objective').textContent = data.objective;
                            document.getElementById('iteration').textContent = data.iteration + ' / ' + data.max_iterations;
                            
                            // Update status badge
                            const statusBadge = document.getElementById('status-badge');
                            statusBadge.textContent = data.status;
                            statusBadge.className = 'status-badge';
                            if (data.status === 'IN_PROGRESS') {
                                statusBadge.classList.add('status-in-progress');
                            } else if (data.status === 'COMPLETED') {
                                statusBadge.classList.add('status-completed');
                            } else {
                                statusBadge.classList.add('status-max-reached');
                            }
                            
                            // Update goal stats
                            document.getElementById('goals-total').textContent = data.goals.total;
                            document.getElementById('goals-completed').textContent = data.goals.statistics.COMPLETED;
                            document.getElementById('goals-failed').textContent = data.goals.statistics.FAILED;
                            document.getElementById('goals-pending').textContent = data.goals.statistics.PENDING;
                            
                            // Update goal table
                            const goalTable = document.getElementById('goal-table');
                            let goalHtml = `<tr><th>ID</th><th>Description</th><th>Status</th><th>Priority</th></tr>`;
                            data.goals.details.forEach(goal => {
                                const statusClass = goal.status === 'COMPLETED' ? 'text-success' : goal.status === 'FAILED' ? 'text-danger' : 'text-secondary';
                                const description = typeof goal.description === 'object' ? JSON.stringify(goal.description) : goal.description;
                                goalHtml += `<tr>
                                    <td>${goal.id}</td>
                                    <td>${description}</td>
                                    <td class="${statusClass}">${goal.status}</td>
                                    <td>${goal.priority}</td>
                                </tr>`;
                            });
                            goalTable.innerHTML = goalHtml;
                            
                            // Update tool list
                            const toolsList = document.getElementById('tools-list');
                            let toolsHtml = '<strong>Base Tools:</strong> ' + data.tools.base_tools.join(', ') + '<br><br>';
                            toolsHtml += '<strong>Custom Tools:</strong> ' + (data.tools.custom_tools.length ? data.tools.custom_tools.join(', ') : 'None');
                            toolsList.innerHTML = toolsHtml;
                            
                            // Update human feedback
                            document.getElementById('human-feedback').textContent = data.human_feedback || 'No feedback provided';
                            
                            // Update last updated timestamp
                            document.getElementById('last-updated').textContent = new Date(data.last_updated).toLocaleString();
                        }
                        
                        // Initial load
                        document.addEventListener('DOMContentLoaded', () => {
                            refreshData();
                            // Refresh every 5 seconds
                            setInterval(refreshData, 5000);
                        });
                    </script>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>Agent Monitoring Dashboard</h1>
                            <p>Objective: <span id="objective">Loading...</span></p>
                            <p>
                                Status: <span id="status-badge" class="status-badge">Loading...</span>
                                Iteration: <span id="iteration">-/-</span>
                                Last updated: <span id="last-updated">-</span>
                            </p>
                        </div>
                        
                        <div class="flex-container">                
                            <div class="flex-item">
                                <div class="card">
                                    <h2>Goal Statistics</h2>
                                    <p>Total Goals: <strong id="goals-total">-</strong></p>
                                    <p>Completed: <strong id="goals-completed" style="color:green">-</strong></p>
                                    <p>Failed: <strong id="goals-failed" style="color:red">-</strong></p>
                                    <p>Pending: <strong id="goals-pending" style="color:gray">-</strong></p>
                                </div>
                                
                                <div class="card">
                                    <h2>Tools</h2>
                                    <div id="tools-list">Loading tools...</div>
                                </div>
                                
                                <div class="card">
                                    <h2>Human Feedback</h2>
                                    <p id="human-feedback">No feedback provided</p>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <h2>Goals</h2>
                            <table id="goal-table">
                                <tr>
                                    <th>ID</th>
                                    <th>Description</th>
                                    <th>Status</th>
                                    <th>Priority</th>
                                </tr>
                                <tr>
                                    <td colspan="4">Loading goals...</td>
                                </tr>
                            </table>
                        </div>
                        
                        <div class="card">
                            <h2>Live Logs</h2>
                            <div id="log-container" class="log-container">
                                <p class="log-item">Loading logs...</p>
                            </div>
                        </div>
                    </div>
                </body>
                </html>
                """
                return html
        
        def run_server():
            port = 7777
            with socketserver.TCPServer(("", port), AgentDashboardHandler) as httpd:
                httpd.agent = self
                self.log(f"Dashboard started at http://localhost:{port}/", "INFO")
                httpd.serve_forever()
        
        # Start server in a separate thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Store thread reference
        self.dashboard_thread = server_thread

    def run(self):
        """Run the autonomous agent main loop with real-time monitoring"""
        self.log(f"Starting agent with objective: {self.objective}")
        self.log(f"Max iterations: {self.max_iterations}")
        
        # Start the monitoring dashboard
        self.start_monitoring_dashboard()
        print(f"\n👀 Real-time monitoring dashboard available at http://localhost:7777/\n")
        
        self.generate_initial_goals()
        self.save_memory()
        
        while self.current_iteration < self.max_iterations and \
            self.agent_memory['system']['status'] == 'IN_PROGRESS':
            
            self.log(f"\n=== Iteration {self.current_iteration + 1}/{self.max_iterations} ===")
            
            user_feedback = None
            # Get human feedback before starting the iteration
            if not self.auto:
                user_feedback = self.get_human_feedback("Give the agent any specific input before the iteration start:")
            
            # Save checkpoint before the iteration
            self.save_checkpoint()
            
            # Process goals and generate new ones
            self.process_goals()
            
            # Check if we should create a new tool
            if self.current_iteration % 2 == 1 and len(self.custom_tools) < 3:  # Limit to 3 custom tools
                pending_goals = [g for g in self.goals if g['status'] == 'PENDING']
                if pending_goals:
                    goal_desc = pending_goals[0]['description']
                    self.create_tool(f"Tool to help with: {goal_desc}", None)
            
            self.current_iteration += 1
            
            iteration_log = {
                "number": self.current_iteration,
                "goals_processed": len([g for g in self.goals if g['status'] != 'PENDING']),
                "pending_goals": len([g for g in self.goals if g['status'] == 'PENDING']),
                "new_tools": len(self.custom_tools),
                "timestamp": datetime.now().isoformat(),
                "user_feedback": user_feedback if user_feedback else ""
            }
            
            self.agent_memory['system']['iterations'].append(iteration_log)
            self.save_memory()
                
            # Check if all goals are completed
            pending_goals = [g for g in self.goals if g['status'] == 'PENDING']
            if not pending_goals and self.agent_memory['system']['status'] != 'COMPLETED':
                # Generate one more set of goals if needed
                prompt = PromptTemplate(
                    template="""
                    Given the objective: {objective}
                    And these completed goals: {completed_goals}
                    
                    Is the objective fully achieved? Answer as a JSON:
                    {{
                        "is_complete": true/false,
                        "reasoning": "explanation",
                        "additional_goals": [] (if not complete)
                    }}
                    """
                )
                
                check_chain = prompt | self.llm | StrOutputParser()
                
                try:
                    check_result = check_chain.invoke({
                        "objective": self.objective,
                        # "human_input": user_feedback,
                        "completed_goals": json.dumps([g for g in self.goals if g['status'] == 'COMPLETED'], indent=2)
                    })
                    
                    # Extract JSON
                    json_match = re.search(r'\{.*\}', check_result.replace('\n', ' '), re.DOTALL)
                    if json_match:
                        completion_data = json.loads(json_match.group())
                        
                        if completion_data.get("is_complete", False):
                            self.agent_memory['system']['status'] = 'COMPLETED'
                            self.log("All goals completed. Objective achieved!")
                        else:
                            # Add new goals if necessary
                            for goal in completion_data.get("additional_goals", []):
                                self._add_goal(goal, 1)
                            
                            if not completion_data.get("additional_goals"):
                                # Fallback
                                self._add_goal("Final check of objective completion", 1)
                except Exception as e:
                    self.log_error(f"Completion check failed: {str(e)}")
                    self._add_goal("Verify objective completion", 1)
        
        if self.agent_memory['system']['status'] != 'COMPLETED':
            self.agent_memory['system']['status'] = 'MAX_ITERATIONS_REACHED'
        
        self.save_memory()
        
        # Final summary
        return self._generate_summary()

    def _generate_summary(self):
        """Generate a summary of the agent's work"""
        completed_goals = [g for g in self.goals if g['status'] == 'COMPLETED']
        failed_goals = [g for g in self.goals if g['status'] == 'FAILED']
        pending_goals = [g for g in self.goals if g['status'] == 'PENDING']
        
        prompt = PromptTemplate(
            template="""
            Generate a comprehensive summary of the work done towards this objective: {objective}
            
            Completed Goals ({completed_goals_count}):
            {completed_goals}
            
            Failed Goals ({failed_goals_count}):
            {failed_goals}
            
            Pending Goals ({pending_goals_count}):
            {pending_goals}
            
            Key Tools Created:
            {tools}
            
            Recent Reflections:
            {reflections}
            
            Provide a concise yet detailed summary covering:
            1. Major accomplishments and discoveries
            2. Key insights gained
            3. Challenges encountered
            4. Overall assessment of objective completion
            5. Recommendations for next steps
            
            Format as a well-organized summary with clear sections.
            """
        )
        
        # Create a runnable chain
        summary_chain = prompt | self.llm | StrOutputParser()
        
        try:
            # Get the latest reflections (up to 3)
            recent_reflections = self.agent_memory['reflections'][-3:] if self.agent_memory['reflections'] else []
            
            # Invoke the chain
            summary = summary_chain.invoke({
                "objective": self.objective,
                "completed_goals": json.dumps(completed_goals, indent=2),
                "completed_goals_count": len(completed_goals),
                "failed_goals": json.dumps(failed_goals, indent=2),
                "failed_goals_count": len(failed_goals),
                "pending_goals": json.dumps(pending_goals, indent=2),
                "pending_goals_count": len(pending_goals),
                "tools": json.dumps(self.agent_memory['tools'], indent=2),
                "reflections": json.dumps(recent_reflections, indent=2)
            })
            
            return summary
        except Exception as e:
            self.log_error(f"Summary generation failed: {str(e)}")
            return f"Summary generation failed: {str(e)}"


    def log(self, message, level="INFO"):
        """Enhanced logging with levels"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if level == "ERROR":
            print(f"[{timestamp}] \033[91mERROR: {message}")  # Red text for errors
        elif level == "WARNING":
            print(f"[{timestamp}] \033[93mWARNING: {message}")  # Yellow text for warnings
        elif self.verbose or level == "ERROR" or level == "WARNING":
            print(f"[{timestamp}] {message}")
        
        # Log to file regardless of verbose setting
        log_dir = os.path.join("./workspace", "logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"agent_log_{datetime.now().strftime('%Y%m%d')}.txt"), "a") as f:
            f.write(f"[{timestamp}] {level}: {message}\n")

    def get_human_feedback(self, message=""):
        """Get structured feedback from human"""
        prompt = message if message else "Enter feedback (or press Enter to continue):"
        print(f"\n{BLUE}{prompt}")  # Blue text for prompt
        
        feedback = input("> ").strip()
        
        if feedback:
            self.human_feedback = feedback
            self.agent_memory['human_feedback'].append({
                "timestamp": datetime.now().isoformat(),
                "feedback": feedback
            })
            self.log(f"Received human feedback: {feedback}", "INFO")
        
        return feedback
    
    def log_error(self, message):
        """Log an error message"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {message}")

    def _extract_code(self, response):
        """Extract code from LLM response"""
        try:
            # Try to extract code from markdown code blocks
            if "```python" in response:
                return response.split("```python")[1].split("```")[0].strip()
            elif "```" in response:
                return response.split("```")[1].strip()
            return response.strip()
        except Exception:
            return response
            

    def execute_tool_manually(self, tool_name, tool_input):
        """Execute a specific tool with given input"""
        try:
            # Find the tool in base tools
            for tool in self.base_tools:
                if tool.name == tool_name:
                    self.log(f"Executing base tool: {tool_name}")
                    result = tool.func(tool_input)
                    return result
            
            # Check in custom tools if not found in base tools
            if tool_name in self.custom_tools:
                self.log(f"Executing custom tool: {tool_name}")
                result = self.custom_tools[tool_name].func(tool_input)
                return result
            
            # If tool not found
            return f"Error: Tool '{tool_name}' not found. Available tools: " + \
                  ", ".join([t.name for t in self.base_tools] + list(self.custom_tools.keys()))
        
        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            self.log_error(error_msg)
            return error_msg
        

if __name__ == "__main__":

    # env
    api_key = os.environ.get("API_KEY")
    model = os.environ.get("MODEL_SLUG")
    api_base = os.environ.get("PROVIDER_BASE_URL")

    if not api_key:
        raise ValueError("API_KEY environment variable not set")
    
    if not model:
        raise ValueError("MODEL_SLUG environment variable not set")
    
    if not api_key:
        raise ValueError("PROVIDER_BASE_URL environment variable not set")

    parser = argparse.ArgumentParser(description='Enhanced Autonomous Agent System with LangChain')
    parser.add_argument('--objective', type=str, required=True, help='The main objective for the agent')
    parser.add_argument('--max_iterations', type=int, default=5, help='Maximum number of iterations')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--auto', action='store_true', help='Run on auto pilot mode no human feedback between iterations')

    args = parser.parse_args()
    
    agent = AutonomousAgent(
        objective=args.objective,
        max_iterations=args.max_iterations,
        model=model,
        verbose=args.verbose,
        api_base=api_base,
        api_key=api_key,
        auto=args.auto
    )
    
    summary = agent.run()
    print("\n=== EXECUTION SUMMARY ===")
    print(summary)