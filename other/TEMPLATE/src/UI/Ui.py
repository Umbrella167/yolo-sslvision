import dearpygui.dearpygui as dpg
import src.UI.Theme as theme
from src.UI.LayoutManager import layoutManager
import src.UTILS.Utils as utils
import cv2
import os
from SAM2.SAM2 import sam2
import numpy as np


class UiData:
    def __init__(self):
        dpg.create_context()

        pass


class DiyComponents:
    def __init__(self, data: UiData):
        pass


class UiCallBack:
    def __init__(self, data: UiData, component: DiyComponents):
        self.select_image = None
        self.points = []
        self.labels = []
        self.image_width = 0
        self.image_height = 0
        self.obj_id = 0
        self.image_list = []
        pass

    def save_layout(self, sender, app_data, user_data):
        if dpg.is_key_down(dpg.mvKey_Control) and dpg.is_key_down(dpg.mvKey_Control):
            layoutManager.save_layout()

    def file_dialog_callback(self, sender, app_data, user_data):
        dpg.set_value(user_data, app_data["file_path_name"])

    def set_itme_text_color(self, item, color):
        with dpg.theme() as input_text:
            with dpg.theme_component(dpg.mvInputText):
                dpg.add_theme_color(dpg.mvThemeCol_Text, color)
            dpg.bind_item_theme(item, input_text)

    def import_vedio(self, sender, app_data, user_data):
        vedio_path = dpg.get_value(user_data)
        # print(vedio_path)
        cap = cv2.VideoCapture(vedio_path)
        if not cap.isOpened():
            print("vedio path error")
            self.set_itme_text_color("vedio_path_inputtext", (255, 0, 0, 220))
            return
        self.set_itme_text_color("vedio_path_inputtext", (0, 255, 0, 220))
        # total_frames , times = utils.get_video_info_ffmpeg(vedio_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rate = int(cap.get(cv2.CAP_PROP_FPS))
        times = utils.format_time(total_frames / rate)
        dpg.set_value("vedio_info_text", f"总帧数:{total_frames},  时长:{times}")

    def generate_images(self, sender, app_data, user_data):
        vedio_path = dpg.get_value("vedio_path_inputtext")
        start_frame = dpg.get_value("start_frame")
        end_frame = dpg.get_value("end_frame")
        step = dpg.get_value("step")
        quality = dpg.get_value("quality")
        utils.generate_images(
            vedio_path, start_frame, end_frame, step, quality, size=(1080, 720)
        )

    def rename(self, sender, app_data, user_data):
        # 把resource/output_images 里的图片从0开始重命名
        output_dir = "resource/output_images"
        if not os.path.exists(output_dir):
            print(f"Directory {output_dir} does not exist.")
            return

        images = sorted(os.listdir(output_dir))
        for i, image in enumerate(images):
            old_path = os.path.join(output_dir, image)
            new_path = os.path.join(output_dir, f"{i:05d}.jpg")
            os.rename(old_path, new_path)
        print("Images have been renamed successfully.")

    def read_images(self):
        dpg.delete_item(item="file_window", children_only=True)
        output_dir = "resource/output_images"
        if not os.path.exists(output_dir):
            print(f"Directory {output_dir} does not exist.")
            return
        images = sorted(os.listdir(output_dir))
        self.image_list = images
        max_index = len(images)
        step_size = 1

        image_path = os.path.join(output_dir, images[0])
        width, height, channels, data = dpg.load_image(image_path)
        for i in range(0, len(images), step_size):
            image = images[i]
            with dpg.group(horizontal=True, parent="file_window"):
                dpg.add_selectable(
                    label=f"{image}",
                    tag=f"{image}_selectable",
                    user_data=images,
                    callback=self.selectable_callback,
                )

        self.image_width = width
        self.image_height = height
        if not dpg.does_alias_exist("raw_texture"):
            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(
                    width=width,
                    height=height,
                    default_value=data,
                    tag="raw_texture",
                    format=dpg.mvFormat_Float_rgb,
                )

        if not dpg.does_alias_exist("raw_texture_select"):
            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(
                    width=width,
                    height=height,
                    default_value=data,
                    tag="raw_texture_select",
                    format=dpg.mvFormat_Float_rgb,
                )

    def selectable_callback(self, sender, app_data, user_data):
        print(user_data)
        for image in user_data:
            item = f"{image}_selectable"
            if item != sender:
                if dpg.does_alias_exist(item):
                    dpg.set_value(item, False)
        sender = dpg.get_item_alias(sender).split("_")[0]
        self.select_image = sender
        image = cv2.imread(f"resource/output_images/{sender}")
        texture = utils.image2texture(image)
        dpg.set_value("raw_texture_select", texture)
        dpg.delete_item("canvas", children_only=True)
        dpg.draw_image(
            texture_tag="raw_texture_select",
            pmin=[0, 0],
            pmax=[self.image_width, self.image_height],
            parent="canvas",
        )

    def next_image(self, sender, app_data):
        if self.select_image is None:
            return
        str_split = self.select_image.split(".")
        index = int(str_split[0])
        str_end = str_split[1]

        if app_data < 0:
            max_index = len(self.image_list) - 1
            index += 1
            index = max_index if index > max_index else index
        else:
            index -= 1
            index = 0 if index < 0 else index

        select_image = f"{index:05d}.{str_end}"
        dpg.set_value(f"{select_image}_selectable", True)
        self.selectable_callback(f"{select_image}_selectable", None, self.image_list)

        # if not dpg.does_alias_exist(f"{select_image}_image"):
        #     return
        image = cv2.imread(f"resource/output_images/{select_image}")
        texture = utils.image2texture(image)
        dpg.set_value("raw_texture_select", texture)

        dpg.delete_item("canvas", children_only=True)
        dpg.draw_image(
            texture_tag="raw_texture_select",
            pmin=[0, 0],
            pmax=[self.image_width, self.image_height],
            parent="canvas",
        )
        self.select_image = f"{index:05d}.{str_end}"

    def show_image(self, image):
        texture = np.array(image).ravel().astype(np.float32) / 255.0
        dpg.set_value("raw_texture", texture)
        dpg.delete_item("canvas", children_only=True)
        dpg.draw_image(
            texture_tag="raw_texture",
            pmin=[0, 0],
            pmax=[self.image_width, self.image_height],
            parent="canvas",
        )
    
    def new_obj(self):
        self.obj_id += 1
        self.clear_points()

    def add_points(self, sender, app_data, user_data):
        mouse_pos = dpg.get_drawing_mouse_pos()
        button = dpg.get_item_user_data(sender)
        if button == "LEFT":
            self.points.append(mouse_pos)
            self.labels.append(1)
            dpg.draw_circle(
                mouse_pos, 8, color=(255, 0, 0, 150), parent="canvas", fill=True
            )
        if button == "RIGHT":
            self.points.append(mouse_pos)
            self.labels.append(0)
            dpg.draw_circle(mouse_pos, 8, color=(0, 0, 255, 150), parent="canvas")
        # self.show_predict()

    def show_predict(self):
        frame_id = int(self.select_image.split(".")[0])
        frame = cv2.imread(f"resource/output_images/{self.select_image}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_obj_ids, out_mask_logits = sam2.add_point(
            points=self.points,
            labels=self.labels,
            frame_id=frame_id,
            obj_id=self.obj_id,
        )
        mask = sam2._logits2mask(out_mask_logits, out_obj_ids)
        # mask = sam2.overlay_mask_on_image
        rect, frame = sam2.mask2rect(mask, frame)
        print("Points: ", self.points, "\nLabels: ", self.labels)
        print("Frame: ", frame_id)
        print("ObjId: ", self.obj_id)
        self.show_image(frame)

    def clear_points(self):
        self.points = []
        self.labels = []

    def generate_yolo_datas(self):
        sam2.generate_yolo_datas()

    def change_obj_id(self,sender, app_data, user_data):
        if app_data < 48 or app_data > 57:
            return
        obj_id = app_data - 48
        self.obj_id = obj_id
        self.clear_points()

    def reset_sam2(self):
        self.points = []
        self.labels = []
        sam2.reset()

class UI:
    def __init__(self):
        self._data = UiData()
        self._diycomponents = DiyComponents(self._data)
        self._callback = UiCallBack(self._data, self._diycomponents)
        theme.set_theme("Dark")
        theme.set_font(15)

    def show_ui(self):
        layoutManager.load_layout()
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def run_loop(self, func=None):
        if func is not None:
            while dpg.is_dearpygui_running():
                func()
                dpg.render_dearpygui_frame()
        else:
            dpg.start_dearpygui()

    def create_global_handler(self):
        with dpg.handler_registry() as global_hander:
            dpg.add_key_release_handler(
                label="save_layout", callback=self._callback.save_layout
            )
            dpg.add_key_release_handler(
                key=dpg.mvKey_F1, callback=self.pop_import_vedio_window
            )
            dpg.add_key_release_handler(
                key=dpg.mvKey_F2, callback=self._callback.read_images
            )
            dpg.add_key_release_handler(
                callback=self._callback.change_obj_id
            )
            dpg.add_key_release_handler(
                key=dpg.mvKey_Return, callback=self._callback.show_predict
            )
            dpg.add_key_release_handler(
                key=dpg.mvKey_R, callback=self._callback.reset_sam2
            )
            dpg.add_key_release_handler(
                key=dpg.mvKey_End, callback=self._callback.generate_yolo_datas
            )
            dpg.add_key_release_handler(
                key=dpg.mvKey_Escape, callback=self._callback.clear_points
            )
            dpg.add_key_release_handler(
                key=dpg.mvKey_Spacebar, callback=self._callback.new_obj
            )
            dpg.add_mouse_wheel_handler(callback=self._callback.next_image)

            dpg.add_mouse_double_click_handler(
                button=dpg.mvMouseButton_Right,
                callback=self._callback.add_points,
                user_data="RIGHT",
            )
            dpg.add_mouse_double_click_handler(
                button=dpg.mvMouseButton_Left,
                callback=self._callback.add_points,
                user_data="LEFT",
            )

    def pop_import_vedio_window(self):
        if dpg.does_alias_exist("pop_inport_vedio_window"):
            dpg.show_item("pop_inport_vedio_window")
            dpg.configure_item("pop_inport_vedio_window", pos=dpg.get_mouse_pos())
            return
        with dpg.window(
            pos=dpg.get_mouse_pos(),
            tag="pop_inport_vedio_window",
            label="import vedio",
            width=300,
            height=300,
            no_resize=True,
        ):
            with dpg.group(horizontal=True):
                dpg.add_text("视频路径:")
                dpg.add_input_text(tag="vedio_path_inputtext", width=-1)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=170)
                dpg.add_button(
                    label="Browse", callback=lambda: dpg.show_item("file_dialog_id")
                )
                dpg.add_button(
                    label="Import",
                    callback=self._callback.import_vedio,
                    user_data="vedio_path_inputtext",
                )
            dpg.add_text(default_value="", tag="vedio_info_text")
            with dpg.child_window(height=-1, width=-1):
                with dpg.group(horizontal=True):
                    dpg.add_text("开始帧:")
                    dpg.add_input_int(tag="start_frame", width=-1)
                with dpg.group(horizontal=True):
                    dpg.add_text("结束帧:")
                    dpg.add_input_int(tag="end_frame", width=-1)
                with dpg.group(horizontal=True):
                    dpg.add_text("步长:")
                    dpg.add_input_int(tag="step", width=-1)
                with dpg.group(horizontal=True):
                    dpg.add_text("质量:")
                    dpg.add_input_int(tag="quality", width=-1)
                dpg.add_spacer(height=5)

                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=150, height=30)
                    dpg.add_button(
                        label="重命名", width=55, callback=self._callback.rename
                    )
                    dpg.add_button(
                        label="导入", width=55, callback=self._callback.generate_images
                    )

            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                tag="file_dialog_id",
                width=700,
                height=400,
                callback=self._callback.file_dialog_callback,
                user_data="vedio_path_inputtext",
            ):
                dpg.add_file_extension(
                    "Source files (*.mp4 *.wmv *.flv *.avi *.webm){.mp4,.wmv,.flv,.avi,.webm}",
                    color=(0, 255, 255, 255),
                )
                dpg.add_file_extension(".*")

    def create_main_window(self):
        with dpg.window(tag="main_window"):
            with dpg.group(horizontal=True):
                with dpg.child_window(height=-1, width=700, tag="file_window"):
                    pass
                with dpg.child_window(height=-1, width=-1, tag="image_window"):
                    with dpg.drawlist(width=-1, height=-1, tag="main_drawlist"):
                        with dpg.draw_node(tag="canvas"):
                            dpg.draw_circle([0, 0], 50)
                            pass
        dpg.set_primary_window("main_window", True)

    def update_main_window(self):
        width, height = dpg.get_item_rect_size("main_window")
        dpg.configure_item("file_window", width=width / 8)

        width, height = dpg.get_item_rect_size("image_window")
        dpg.set_item_width("main_drawlist", width)
        dpg.set_item_height("main_drawlist", height - 20)

    def create_viewport(self, lable: str = "", width: int = 1920, height: int = 1080):
        self.create_global_handler()
        dpg.configure_app(
            docking=False,
            docking_space=False,
            init_file=layoutManager.dpg_window_config,
            load_init_file=True,
        )
        dpg.create_viewport(title=lable, width=width, height=height)


ui = UI()
