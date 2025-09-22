import sys
import tempfile

from llmring.mcp.client.mcp_client import MCPClient


def _minimal_stdio_server_code() -> str:
    return (
        "import sys, json\n"
        "def respond(resp):\n"
        "    sys.stdout.write(json.dumps(resp) + '\\n')\n"
        "    sys.stdout.flush()\n"
        "while True:\n"
        "    line = sys.stdin.readline()\n"
        "    if not line:\n"
        "        break\n"
        "    try:\n"
        "        msg = json.loads(line)\n"
        "        id_ = msg.get('id')\n"
        "        method = msg.get('method')\n"
        "        if method == 'initialize':\n"
        "            respond({'jsonrpc':'2.0','id':id_,'result':{'protocolVersion':'2025-06-18','serverInfo':{'name':'min','version':'1.0.0'},'capabilities':{'tools':{'list':True,'call':True},'prompts':{'list':True,'get':True},'resources':{'list':True,'read':True}}}})\n"
        "        elif method == 'tools/list':\n"
        "            respond({'jsonrpc':'2.0','id':id_,'result':{'tools':[]}})\n"
        "        elif method == 'roots/list':\n"
        "            respond({'jsonrpc':'2.0','id':id_,'result':{'roots':[]}})\n"
        "        else:\n"
        "            respond({'jsonrpc':'2.0','id':id_,'error':{'code':-32601,'message':'Method not found'}})\n"
        "    except Exception as e:\n"
        "        respond({'jsonrpc':'2.0','id':None,'error':{'code':-32700,'message':'Parse error','data':str(e)}})\n"
    )


def test_minimal_stdio_server_initialize_and_tools():
    code = _minimal_stdio_server_code()
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as f:
        f.write(code)
        path = f.name

    client = MCPClient.stdio(
        command=[sys.executable, "-u", path], timeout=10.0, allow_unsafe_commands=True
    )
    try:
        with client:
            res = client.initialize()
            assert isinstance(res, dict)
            tools = client.list_tools()
            assert isinstance(tools, list)
    finally:
        pass
