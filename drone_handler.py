from djitellopy import tello
import time
import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DroneHandler:
    """Handles connection, control, and video streaming for the Tello drone."""

    def __init__(self, max_retry=3, retry_delay=5):
        self.drone = tello.Tello()
        self.is_connected = False
        self.is_streaming = False
        self.frame_reader = None
        self.max_retry = max_retry
        self.retry_delay = retry_delay # seconds
        self.is_flying = False # Basic flag to track takeoff state

    def connect(self):
        """Connects to the drone with retries."""
        attempt = 0
        while attempt < self.max_retry and not self.is_connected:
            attempt += 1
            logging.info(f"Connecting to drone (Attempt {attempt}/{self.max_retry})...")
            try:
                self.drone.connect()
                # self.drone.query_wifi_signal_noise_ratio() # Check connection quality
                self.is_connected = True
                logging.info("Drone connected successfully.")
                print(f"Battery: {self.drone.get_battery()}%")
                return True
            except Exception as e:
                logging.error(f"Failed to connect to drone: {e}")
                if attempt < self.max_retry:
                    logging.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logging.error("Max connection attempts reached.")
                    return False
        return self.is_connected

    def start_stream(self):
        """Starts the video stream."""
        if not self.is_connected:
            logging.warning("Cannot start stream: Drone not connected.")
            return False
        if self.is_streaming:
            logging.info("Stream already started.")
            return True
        try:
            logging.info("Starting video stream...")
            self.drone.streamon()
            self.frame_reader = self.drone.get_frame_read()
             # Allow time for stream buffer to fill
            time.sleep(1)
            # Check if frame reader is working
            if self.frame_reader.frame is None:
                 logging.error("Failed to get initial frame from stream.")
                 self.drone.streamoff()
                 return False
            self.is_streaming = True
            logging.info("Video stream started.")
            return True
        except Exception as e:
            logging.error(f"Failed to start video stream: {e}")
            # Attempt cleanup if streamon succeeded but frame reading failed
            try:
                self.drone.streamoff()
            except Exception:
                 pass # Ignore errors during cleanup
            self.frame_reader = None
            return False

    def get_frame(self, retry_count=3):
        """Gets the current video frame with retries for None frames."""
        if not self.is_streaming or self.frame_reader is None:
            # logging.warning("Cannot get frame: Stream not started or reader not available.")
            return None
        attempts = 0
        while attempts < retry_count:
            try:
                frame = self.frame_reader.frame
                if frame is not None:
                    return frame # BGR format
                else:
                    logging.warning(f"Received None frame (Attempt {attempts+1}/{retry_count}). Waiting briefly...")
                    time.sleep(0.1) # Short pause before retrying
            except Exception as e:
                logging.error(f"Error getting frame: {e}")
                # Consider more robust error handling, e.g., trying to restart stream
                return None # Return None on exception
            attempts += 1
        logging.error(f"Failed to get a valid frame after {retry_count} attempts.")
        return None


    def takeoff(self):
        """Commands the drone to take off."""
        if not self.is_connected:
            logging.warning("Cannot take off: Drone not connected.")
            return False
        if self.is_flying:
            logging.info("Drone is already flying.")
            return True

        logging.info("Taking off...")
        try:
            # Check battery before takeoff
            if self.drone.get_battery() < 15:
                 logging.error(f"Takeoff aborted: Battery low ({self.drone.get_battery()}%)")
                 return False

            response = self.drone.takeoff()
            # takeoff() might not return a boolean or might throw exception on failure
            # Add a small delay and check height as a basic confirmation
            time.sleep(3) # Allow time for takeoff sequence
            if self.drone.get_height() > 0: # Check if it's airborne
                logging.info("Takeoff successful.")
                self.is_flying = True
                return True
            else:
                 logging.warning("Takeoff command sent, but drone doesn't appear to be airborne. Check drone status.")
                 # It might be taking off slowly, or failed. Let's assume failure for now.
                 # Attempt landing just in case it's stuck in a weird state
                 try:
                     self.drone.land()
                 except Exception:
                     pass
                 return False

        except Exception as e:
            logging.error(f"Takeoff failed: {e}")
            return False

    def land(self):
        """Commands the drone to land."""
        if not self.is_connected:
            logging.warning("Cannot land: Drone not connected.")
            return False
        if not self.is_flying:
             logging.info("Drone is not flying.")
             # Ensure RC commands are zeroed if landing from ground state
             self.hover()
             return True

        logging.info("Landing...")
        try:
            response = self.drone.land()
            self.is_flying = False # Assume landing initiated
            # Landing takes time, don't wait indefinitely here
            logging.info("Landing command sent.")
             # Consider adding a short sleep if subsequent actions require landing completion
            # time.sleep(5)
            return True
        except Exception as e:
            logging.error(f"Landing failed: {e}")
            # State is uncertain, might still be flying
            return False

    def hover(self):
        """Commands the drone to hover in place by sending zero movement commands."""
        if not self.is_connected:
            logging.warning("Cannot hover: Drone not connected.")
            return False
        # Sending all zeros tells the drone to maintain current position
        # This is crucial especially if previous RC commands were sent
        # logging.debug("Sending hover command (RC 0,0,0,0).")
        try:
            self.drone.send_rc_control(0, 0, 0, 0)
            return True
        except Exception as e:
            logging.error(f"Failed to send hover command: {e}")
            return False

    def emergency_stop(self):
        """Initiates emergency stop."""
        logging.critical("EMERGENCY STOP ACTIVATED")
        try:
            self.drone.emergency()
            self.is_flying = False # Assume motors stopped
            self.is_connected = False # Connection might be unstable after emergency
            logging.info("Emergency stop command sent.")
        except Exception as e:
            logging.error(f"Failed to send emergency stop command: {e}")


    def disconnect(self):
        """Cleans up resources and disconnects from the drone."""
        logging.info("Disconnecting drone procedure initiated...")
        # Ensure drone attempts to land if it was flying
        if self.is_flying and self.is_connected:
            logging.warning("Drone was flying during disconnect. Attempting to land first.")
            self.land()
            time.sleep(5) # Give it time to land

        if self.is_streaming:
            logging.info("Stopping video stream...")
            try:
                self.drone.streamoff()
                logging.info("Video stream stopped.")
            except Exception as e:
                logging.error(f"Error stopping stream during disconnect: {e}")

        if self.is_connected:
            logging.info("Closing drone connection...")
            try:
                self.drone.end() # Close the Tello connection
                logging.info("Drone connection closed.")
            except Exception as e:
                logging.error(f"Error ending drone connection during disconnect: {e}")

        self.is_connected = False
        self.is_streaming = False
        self.frame_reader = None
        self.is_flying = False
        logging.info("Drone disconnected.")

# Example usage (for testing this module)
if __name__ == '__main__':
    print("--- Drone Handler Test ---")
    handler = DroneHandler()

    if not handler.connect():
        print("Failed to connect to drone. Exiting.")
        exit()

    if not handler.start_stream():
        print("Failed to start video stream. Disconnecting.")
        handler.disconnect()
        exit()

    print("Drone connected and stream started.")
    print("Controls:")
    print("  T: Takeoff")
    print("  L: Land")
    print("  H: Hover (explicitly send hover command)")
    print("  E: Emergency Stop (!!!)")
    print("  Q: Land and Quit")
    print("Displaying video feed...")

    keep_running = True
    try:
        while keep_running:
            frame = handler.get_frame()
            if frame is not None:
                # Display battery level on frame
                batt = handler.drone.get_battery()
                cv2.putText(frame, f"Batt: {batt}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                 # Display flying status
                status = "FLYING" if handler.is_flying else "LANDED"
                cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Drone Feed", frame)
            else:
                print("Warning: Failed to get frame.")
                # Decide action: continue, break, attempt reconnect?
                # For testing, let's just continue for now
                pass

            # Keep drone hovering if flying
            if handler.is_flying:
                handler.hover()

            key = cv2.waitKey(30) & 0xFF # Increased wait time slightly

            if key == ord('q'):
                print("Q pressed: Landing and quitting...")
                keep_running = False
                handler.land() # Initiate landing
                time.sleep(5) # Wait for landing before disconnect
            elif key == ord('t'):
                print("T pressed: Taking off...")
                handler.takeoff()
            elif key == ord('l'):
                print("L pressed: Landing...")
                handler.land()
            elif key == ord('h'):
                print("H pressed: Sending hover command...")
                handler.hover()
            elif key == ord('e'):
                print("E pressed: EMERGENCY STOP!")
                handler.emergency_stop()
                keep_running = False # Exit after emergency

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        print("Initiating emergency landing and disconnect.")
        handler.emergency_stop() # Use emergency in case of unexpected errors
    finally:
        # Ensure cleanup happens
        print("Cleaning up...")
        cv2.destroyAllWindows()
        handler.disconnect()
        print("--- Test Finished ---") 