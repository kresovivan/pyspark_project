import socket
from pyspark.sql import SparkSession

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

local_ip = get_local_ip()
print(f"Обнаружен IP-адрес: {local_ip}")

spark = SparkSession.builder \
    .appName("SparkUI") \
    .config("spark.driver.host", local_ip) \
    .config("spark.driver.bindAddress", "0.0.0.0") \
    .config("spark.ui.port", "4040") \
    .master("local[*]") \
    .getOrCreate()

print(f"✅ Spark UI доступен по адресу: http://{local_ip}:4040")
print("📊 Откройте эту ссылку в браузере")
print("⏸️ Нажмите Enter для остановки Spark и выхода...")

# Ждём ввода пользователя, чтобы программа не завершалась
input()

spark.stop()
print("👋 Spark остановлен")