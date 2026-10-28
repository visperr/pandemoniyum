"""
Keeps the tracker running at full speed while the game is in front (macOS only).

With the game fullscreen, the tracker is a background process: macOS may put it in App Nap (throttled timers,
lower CPU priority) and treat it as background work, so tracking slows right down exactly when it's needed.
This tells macOS the tracker is doing latency-critical, user-initiated work, which keeps it out of App Nap,
and runs it at interactive priority. Plain ctypes calls into Foundation, so no extra packages are needed.
"""
import ctypes
import ctypes.util
import sys

# NSActivityOptions (Foundation/NSProcessInfo.h)
_NS_ACTIVITY_USER_INITIATED = 0x00FFFFFF       # also keeps the Mac from idle-sleeping mid-game
_NS_ACTIVITY_LATENCY_CRITICAL = 0xFF00000000

# qos_class_t (sys/qos.h)
_QOS_CLASS_USER_INTERACTIVE = 0x21

_activity = None  # held for the life of the process; the activity ends if it's released


def keep_tracker_responsive() -> bool:
    """
    Call once at startup, before any threads are created (threads inherit the priority).
    Returns True if it took effect; False on other systems or if the calls aren't available.
    """
    global _activity
    if _activity is not None:
        return True
    if sys.platform != "darwin":
        return False
    try:
        libsystem = ctypes.CDLL(ctypes.util.find_library("System"))
        libsystem.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]
        libsystem.pthread_set_qos_class_self_np(_QOS_CLASS_USER_INTERACTIVE, 0)

        objc = ctypes.CDLL(ctypes.util.find_library("objc"))
        ctypes.CDLL(ctypes.util.find_library("Foundation"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        # objc_msgSend has to be called through a prototype matching each message's arguments (arm64)
        send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        send_plain = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(send)
        send_cstring = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(send)
        send_activity = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_uint64, ctypes.c_void_p)(send)

        process_info = send_plain(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
        reason = send_cstring(objc.objc_getClass(b"NSString"), objc.sel_registerName(b"stringWithUTF8String:"),
                              b"Live motion tracking for PANdemoniYUM")
        activity = send_activity(process_info, objc.sel_registerName(b"beginActivityWithOptions:reason:"),
                                 _NS_ACTIVITY_USER_INITIATED | _NS_ACTIVITY_LATENCY_CRITICAL, reason)
        if not activity:
            return False
        send_plain(activity, objc.sel_registerName(b"retain"))
        _activity = activity
        return True
    except (OSError, AttributeError, TypeError):
        return False
