# your_app_name/management/commands/clear_media.py

from django.core.management.base import BaseCommand
import os
import shutil
from django.conf import settings

class Command(BaseCommand):
    help = 'Deletes all files in the media folder'

    def handle(self, *args, **kwargs):
        media_root = settings.MEDIA_ROOT
        print(media_root)
        if os.path.isdir(media_root):
            for filename in os.listdir(media_root):
                file_path = os.path.join(media_root, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    self.stdout.write(self.style.SUCCESS(f'Deleted {file_path}'))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f'Failed to delete {file_path}. Reason: {e}'))
        else:
            self.stdout.write(self.style.ERROR(f'{media_root} is not a valid directory'))
