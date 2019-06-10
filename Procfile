release: python manage.py migrate --noinput
web: gunicorn OnlineReputationAnalyzer.wsgi --timeout 60
clock: python clock.py