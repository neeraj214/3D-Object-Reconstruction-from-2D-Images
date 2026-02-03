
try:
    from server import api
    print("server.api imported successfully")
except Exception as e:
    print(f"Error importing server.api: {e}")
    import traceback
    traceback.print_exc()
