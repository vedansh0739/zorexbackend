# your_app/management/commands/custom_runsslserver.py

from sslserver.management.commands.runsslserver import Command as SSLServerCommand
import os
import shutil
from django.conf import settings

class Command(SSLServerCommand):
    def handle(self, *args, **kwargs):
        self.clear_media_folder()
        super().handle(*args, **kwargs)

    def clear_media_folder(self):
        media_root = settings.MEDIA_ROOT
        if os.path.isdir(media_root):
            for filename in os.listdir(media_root):
                file_path = os.path.join(media_root, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            self.stdout.write(self.style.SUCCESS('Cleared media folder.'))
        else:
            self.stdout.write(self.style.ERROR('Media folder not found.'))
