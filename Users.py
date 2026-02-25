import hashlib
import os
import pickle


class User:
    def __init__(self, username, password, is_admin=False):
        self.username = username
        self.password = self.hash_password(password)
        self.is_admin = is_admin

    @staticmethod
    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()

    def check_password(self, password, saved):
        if saved is True:
            return True
        else:
            return self.password == self.hash_password(password)

    def get_username(self):
        return self.username


class UserManager:
    def __init__(self, db_path='usersdb'):
        self.db_path = db_path
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                self.users = pickle.load(f)
        else:
            self.users = {}

    def save_users(self):
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.users, f)

    def add_user(self, username, password, is_admin=False):
        if username in self.users:
            raise Exception('用户已存在')
        self.users[username] = User(username, password, is_admin)
        self.save_users()

    def change_password(self, username, new_password):
        if username not in self.users:
            raise Exception('用户不存在')
        elif self.users[username].password == new_password:
            raise Exception('新密码不能与旧密码相同')
        self.users[username].password = self.users[username].hash_password(new_password)
        self.save_users()

    def check_password(self, username, password, saved=False):
        if username not in self.users:
            raise Exception('用户不存在')
        return self.users[username].check_password(password, saved)

    def check_admin(self, username):
        if username not in self.users:
            raise Exception('用户不存在')
        return self.users[username].is_admin

    def set_admin(self, username, is_admin):
        if username not in self.users:
            raise Exception('用户不存在')
        self.users[username].is_admin = is_admin
        self.save_users()

    def delete_user(self, username):
        if username not in self.users:
            raise Exception('用户不存在')
        if username == 'superadmin':
            raise Exception('超级管理员账户不能删除')
        del self.users[username]
        self.save_users()

    def get_users(self):
        return self.users


if __name__ == '__main__':
    manager = UserManager()
    manager.add_user('superadmin', 'imadmin', is_admin=True)
    manager.save_users()
