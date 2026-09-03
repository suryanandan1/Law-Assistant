import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'leave_policy_project.settings')
application = get_asgi_application()
