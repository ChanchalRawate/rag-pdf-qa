
function NavBar() {
  const handleLogout = () => {
    localStorage.removeItem("token");
    window.location.reload();
  };

  return (
    <div className="navbar">
      <div className="brand">
        <div className="brand-icon">📄</div>

        <span>PDF AI Assistant</span>
      </div>

      <button className="logout-button" onClick={handleLogout}>
        <span className="logout-icon">↪</span>
        Logout
      </button>
    </div>
  );
}

export default NavBar;
