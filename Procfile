release: python manage.py migrate --noinput
web: daphne  d.asgi:application --port $PORT --bind 0.0.0.0 -v2
clock: python clock.py