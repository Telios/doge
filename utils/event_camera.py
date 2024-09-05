import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import threading

class EventCamera:
    def __init__(self, rgb_background=(125, 125, 125),
                       rgb_pos=(255, 255, 255),
                       rgb_neg=(0, 0, 0),
                       fps=100,
                       downscale=False):
        self.capture = dv.io.CameraCapture()
        if not self.capture.isEventStreamAvailable():
            raise RuntimeError("Input camera does not provide an event stream.")
        
        self.capture.setDVSBiasSensitivity(dv.io.CameraCapture.BiasSensitivity.Low)
        self.capture.setDVXplorerEFPS(dv.io.CameraCapture.DVXeFPS.EFPS_CONSTANT_500)

        self.fps = fps
        self.downscale = downscale
        self.visualizer = dv.visualization.EventVisualizer(self.capture.getEventResolution())
        self.visualizer.setBackgroundColor(rgb_background)
        self.visualizer.setPositiveColor(rgb_pos)
        self.visualizer.setNegativeColor(rgb_neg)

        self.slicer = dv.EventStreamSlicer()
        self.slicer.doEveryTimeInterval(timedelta(milliseconds=1000/self.fps), self.slicing_callback)
        self.current_image = None

    def slicing_callback(self, events: dv.EventStore):
        img = self.visualizer.generateImage(events)
        if self.downscale:
            img = cv.resize(img, (85, 64), interpolation=cv.INTER_NEAREST)
            img = img[0:64, 10:74]
        self.current_image = img
        
    def get_current_image(self):
        return self.current_image
    
    def run(self):
        while self.capture.isRunning():
            events = self.capture.getNextEventBatch()

            if events is not None:
                self.slicer.accept(events)

if __name__ == "__main__":
    event_cam = EventCamera(rgb_background=(125, 125, 125),
                            rgb_pos=(255, 255, 255),
                            rgb_neg=(0, 0, 0),
                            fps=100,
                            downscale=True)
                            
    # Start the event camera thread
    event_thread = threading.Thread(target=event_cam.run)
    event_thread.start()
    while True:
        if event_cam.get_current_image() is not None:
            cv.imshow("Event Camera", event_cam.get_current_image())
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    event_thread.join()