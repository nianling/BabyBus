# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'

import smtplib
from email.mime.text import MIMEText
from email.header import Header


class EmailSender:
    def __init__(self, config):
        """
        初始化邮件发送配置
        :param config: dict，发件人、密码、SMTP信息
        """
        self.sender = config['sender']
        self.password = config['password']

        # 默认qq邮箱---QQ邮箱可以QQ和微信提醒
        self.smtp_server = config.get('smtp_server', 'smtp.qq.com')
        self.smtp_port = config.get('smtp_port', 465)

    def _build_email(self, subject, content, receiver):
        """构造邮件内容"""
        message = MIMEText(content, 'plain', 'utf-8')
        message['From'] = Header(self.sender)
        message['To'] = Header(receiver)
        message['Subject'] = Header(subject)
        return message

    def send_email(self, subject, content, receiver):
        """
        发送邮件
        :param subject: 邮件标题
        :param content: 振文内容
        :param receiver: 收件人
        :return:
        """
        if not receiver:
            print("收件人为空,忽略")
            return
        message = self._build_email(subject, content, receiver)
        try:
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as smtp_obj:
                smtp_obj.login(self.sender, self.password)
                smtp_obj.sendmail(self.sender, [receiver], message.as_string())
                print(f"邮件《{subject}》发送给 {receiver}成功！")
        except Exception as e:
            print(f"邮件《{subject}》发送给 {receiver}失败 :", str(e))
