import dearpygui.dearpygui as dpg
import src.UI.Theme as theme
from src.UI.LayoutManager import layoutManager
import src.UTILS.Utils as utils
import cv2
import os
import numpy as np
from auto_mark import randomRules

class UiData:
    def __init__(self):
        dpg.create_context()
        self.sacle = 1
        pass


class DiyComponents:
    def __init__(self, data: UiData):
        pass


class UiCallBack:
    def __init__(self, data: UiData, component: DiyComponents):
        self._data = data
        self.select_image = None
        self.points = []
        self.point = []

        self.labels = []
        self.image_width = 400
        self.image_height = 300
        self.obj_id = 0
        self.image_list = []
        self.image = None
        self.image_copy = None
        self.yolo_data = []
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
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_width, self.image_height))
        data = utils.image2texture(image)

        for i in range(0, len(images), step_size):
            image = images[i]
            with dpg.group(horizontal=True, parent="file_window"):
                dpg.add_selectable(
                    label=f"{image}",
                    tag=f"{image}_selectable",
                    user_data=images,
                    callback=self.selectable_callback,
                )

        if not dpg.does_alias_exist("raw_texture"):
            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(
                    width=self.image_width,
                    height=self.image_height,
                    default_value=data,
                    tag="raw_texture",
                    format=dpg.mvFormat_Float_rgb,
                )

        if not dpg.does_alias_exist("raw_texture_select"):
            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(
                    width=self.image_width,
                    height=self.image_height,
                    default_value=data,
                    tag="raw_texture_select",
                    format=dpg.mvFormat_Float_rgb,
                )

    def selectable_callback(self, sender, app_data, user_data):
        for image in user_data:
            
            item = f"{image}_selectable"
            if item != sender:
                if dpg.does_alias_exist(item):
                    dpg.set_value(item, False)
        sender = dpg.get_item_alias(sender).split("_")[0]
        self.select_image = sender
        image = cv2.imread(f"resource/output_images/{sender}")
        image = cv2.resize(image, (self.image_width, self.image_height))
        self.image = image
        self.image_copy = self.image.copy()

        randomRules.load_image(self.image)

        texture = utils.image2texture(image)
        dpg.set_value("raw_texture_select", texture)
        dpg.delete_item("canvas", children_only=True)
        dpg.draw_image(
            texture_tag="raw_texture_select",
            pmin=[0, 0],
            pmax=[self.image_width, self.image_height],
            parent="canvas",
        )

    def next_image(self, sender, app_data,user_data):

        if dpg.is_key_down(dpg.mvKey_Control) or user_data == "auto":
            
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
            image = cv2.resize(image, (self.image_width, self.image_height))
            self.image = image
            self.image_copy = self.image.copy()
            randomRules.load_image(self.image)
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
        else:
            if app_data > 0:
                self._data.sacle += 0.2
            else:
                self._data.sacle -= 0.2
            self._data.sacle = max(self._data.sacle,0.5)
            scale_matrix = dpg.create_scale_matrix([self._data.sacle,self._data.sacle,self._data.sacle])
            dpg.apply_transform("canvas", scale_matrix)

    
    def new_obj(self):
        self.obj_id += 1
        self.points = []
        self.labels = []
    
    def generate_yolo_data(self):
        yolo_data = self.yolo_data
        image = self.image
        image_name = self.select_image
        labels_name = image_name.split(".")[0] + ".txt"
        output_image_dir = "resource/COLORDATA/output_images/"
        output_labels_dir = "resource/COLORDATA/output_labels/"

        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        image_path = os.path.join(output_image_dir, image_name)
        labels_path = os.path.join(output_labels_dir, labels_name)
        cv2.imwrite(image_path, image)
        with open(labels_path, "w") as label_file:
            for label in yolo_data:
                label_file.write(" ".join(map(str, label)) + "\n")
        self.next_image(0,-1,"auto")
        print(f"save {labels_name} success")
        self.yolo_data = []
    def add_points(self, sender, app_data, user_data):
        
        mouse_pos = dpg.get_drawing_mouse_pos()
        mouse_pos = [int(mouse_pos[0] / self._data.sacle), int(mouse_pos[1] / self._data.sacle)]
        self.point = mouse_pos
        self.points.append(mouse_pos)
        button = dpg.get_item_user_data(sender)
        if button == "LEFT":
            self.labels.append(1)
            dpg.draw_circle(
                mouse_pos, 3, color=(255, 0, 0, 255), parent="canvas", fill=(255, 0, 0, 255)
            )
        if button == "RIGHT":
            self.labels.append(0)
            dpg.draw_circle(mouse_pos, 8, color=(0, 0, 255, 150), parent="canvas")
        
        mask = randomRules.magic_mask_sam2(self.image, self.points[-1])
        mask_width, mask_height = mask.shape
        self.image_copy = cv2.resize(mask,(int(mask_width * self._data.sacle),int(mask_height * self._data.sacle)))
        rect,image,pmin,pmax,obj_id = randomRules.sam2.mask2rect(mask, self.image)
        self.obj_id = obj_id
        _color_list = [(0,0,255,255),(255,255,0,255),(0,255,0,255),(255,0,255,255)]
        dpg.draw_rectangle(pmax=pmax,pmin=pmin,color=_color_list[obj_id],parent="canvas")
        print([self.obj_id] + rect)
        self.yolo_data.append([self.obj_id] + rect)
    def change_threshold(self,sender, app_data, user_data):
        threshold = app_data
        mask = randomRules.magic_mask(self.image, self.point, threshold, 8)
        red_layer = np.zeros_like(self.image)
        red_layer[:, :, 2] = 255

        # Apply the mask to the red layer
        red_masked = cv2.bitwise_and(red_layer, red_layer, mask=mask)

        # Directly add the red_masked layer to the original image
        layer = cv2.add(self.image, red_masked)
        layer_width, layer_height, _ = layer.shape
        self.image_copy = cv2.resize(layer,(int(layer_width * self._data.sacle),int(layer_height * self._data.sacle)))

    def clear_points_draw(self):
        dpg.delete_item("canvas", children_only=True)
        texture = utils.image2texture(self.image)
        dpg.set_value("raw_texture_select", texture)
        dpg.draw_image(
                texture_tag="raw_texture_select",
                pmin=[0, 0],
                pmax=[self.image_width, self.image_height],
                parent="canvas",
            )
    def clear_points(self):
        self.clear_points_draw()
        self.points = []
        self.labels = []
        self.yolo_data = []
    def change_obj_id(self,sender, app_data, user_data):
        if app_data < 48 or app_data > 57:
            return
        obj_id = app_data - 49
        self.obj_id = obj_id
        self.points = []
        self.labels = []
        print(obj_id)
        
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
                key=dpg.mvKey_Escape, callback=self._callback.clear_points
            )
            # dpg.add_key_release_handler(
            #     key=dpg.mvKey_Spacebar, callback=self._callback.new_obj
            # )
            dpg.add_key_release_handler(
                key=dpg.mvKey_Spacebar, callback=self._callback.generate_yolo_data
            )
            dpg.add_mouse_wheel_handler(callback=self._callback.next_image)

            dpg.add_mouse_click_handler(
                button=dpg.mvMouseButton_Right,
                callback=self._callback.add_points,
                user_data="RIGHT",
            )
            dpg.add_mouse_click_handler(
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
            with dpg.child_window(height=120, width=-1, tag="console"):
                    dpg.add_slider_int(min_value = 0,max_value=255,default_value=127,clamped=True,tag="threshold_slider",callback=self._callback.change_threshold)
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
        if self._callback.image_copy is not None:
            self._callback.image_copy = cv2.resize(self._callback.image_copy,(int(self._callback.image_width * self._data.sacle),int(self._callback.image_height * self._data.sacle)))
            cv2.imshow("res",self._callback.image_copy)
            cv2.waitKey(1)

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
