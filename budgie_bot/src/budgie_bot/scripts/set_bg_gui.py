import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import tkinter as tk
from tkinter import messagebox


class SetBGClient(Node):
    def __init__(self):
        super().__init__('set_bg_client')
        self.service_prefix = '/set_background_'
        self.service_type = 'std_srvs/srv/Trigger'
        self.latest_services = []

    def get_matching_services(self):
        services = self.get_service_names_and_types()
        return sorted([
            name for name, types in services
            if self.service_type in types and name.startswith(self.service_prefix)
        ])

    def send_request(self, service_name: str):
        client = self.create_client(Trigger, service_name)
        if not client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f"Service {service_name} not available.")
        req = Trigger.Request()
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def create_gui(node: SetBGClient):
    root = tk.Tk()
    root.title("Set Background Controls")

    tk.Label(root, text="Available Cameras").pack(pady=(10, 5))

    button_frame = tk.Frame(root)
    button_frame.pack(padx=10, pady=5)

    def refresh_buttons():
        for widget in button_frame.winfo_children():
            widget.destroy()

        services = node.get_matching_services()
        node.latest_services = services

        if not services:
            tk.Label(button_frame, text="No camera services found.").pack()
        else:
            for svc in services:
                cam_id = svc.split('_')[-1]
                btn = tk.Button(button_frame, text=f"Reset BG - Camera {cam_id}",
                                command=lambda s=svc: call_service(s))
                btn.pack(fill='x', pady=2)

        root.after(5000, refresh_buttons)  # refresh every 5 sec

    def call_service(service_name):
        try:
            result = node.send_request(service_name)
            if result.success:
                messagebox.showinfo("Success", result.message)
            else:
                messagebox.showerror("Failed", result.message)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    refresh_buttons()
    root.mainloop()


def main():
    rclpy.init()
    node = SetBGClient()
    try:
        create_gui(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
