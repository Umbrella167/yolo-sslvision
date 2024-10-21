from src.UI.Ui import ui
def loop():
    ui.update_main_window()

if __name__ == "__main__":
    ui.create_viewport()
    ui.create_main_window()
    ui.show_ui()
    ui.run_loop(loop)