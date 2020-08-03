<%@page import="action.dbcon"%>
<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.PreparedStatement"%>

<%@page import="java.sql.Connection"%>
<%@page contentType="text/html" pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>CBSE</title>
    </head>
    <body>
        <%
               Connection con = dbcon.connect();
               String name=request.getParameter("name");
               String password=request.getParameter("password");
                if (name.equals("admin") && password.equals("admin")) {
                    out.println("<script>window.alert('Login successfully'); window.location.href='video.jsp'; </script>");
                } else {
                    out.println("<script>window.alert('Login failed'); window.location.href='login.jsp'; </script>");
                }
            
        %>
    </body>
</html>
