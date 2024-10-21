import dearpygui.dearpygui as dpg
import dearpygui.demo as demo

dpg.create_context()
dpg.create_viewport(title='Color picker', width=1280, height=720)
dpg.setup_dearpygui()

with dpg.window(label='Color picker',tag="main_window"):
    dpg.add_color_picker()
    pass
dpg.set_primary_window("main_window", True)
dpg.show_viewport()
if dpg.is_dearpygui_running():
    dpg.start_dearpygui()
dpg.destroy_context()