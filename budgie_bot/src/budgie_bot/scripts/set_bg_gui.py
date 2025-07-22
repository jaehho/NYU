import rclpy
from rclpy.node import Node
from custom_interfaces.srv import SetBG

import tkinter as tk
from tkinter import messagebox


class SetBGClient(Node):
    def __init__(self):
        super().__init__('set_bg_client')
        self.cli = self.create_client(SetBG, 'set_background')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

    def send_request(self, cam_id: str):
        req = SetBG.Request()
        req.camera_id = cam_id
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def create_gui(client_node):
    def call_service():
        cam_id = entry.get().strip()
        if not cam_id:
            messagebox.showwarning("Input Error", "Please enter a camera ID.")
            return

        try:
            result = client_node.send_request(cam_id)
            if result.success:
                messagebox.showinfo("Success", result.message)
            else:
                messagebox.showerror("Failure", result.message)
        except Exception as e:
            messagebox.showerror("Error", f"Service call failed: {e}")

    root = tk.Tk()
    root.title("Set Background")

    tk.Label(root, text="Camera ID:").grid(row=0, column=0, padx=10, pady=10)
    entry = tk.Entry(root)
    entry.grid(row=0, column=1, padx=10)

    tk.Button(root, text="Set BG", command=call_service).grid(row=1, column=0, columnspan=2, pady=10)

    root.mainloop()


def main():
    rclpy.init()
    client_node = SetBGClient()

    try:
        create_gui(client_node)
    finally:
        client_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
